import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from pipeline import CausalDiffusionInferencePipeline
from utils.config import normalize_config, wan_default_config
from utils.inference_utils import load_generator_checkpoint, place_vae_for_streaming, save_video


DEFAULT_SCENE_CUT_PREFIX = "The scene transitions. "
FAR_SHOT_CUT_RE = re.compile(r"^\s*\[shot cut\]\s*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch multi-shot LongLive inference.")
    parser.add_argument("--prompts_path", default="eval_caption_multishot_t2v_100_longlive.json")
    parser.add_argument("--config_path", default="configs/inference.yaml")
    parser.add_argument("--checkpoint_path", required=True, help="Merged generator checkpoint, e.g. LongLive-2.0-5B/model_bf16.pt")
    parser.add_argument("--output_dir", default="videos/multishot_batch")
    parser.add_argument("--gpus", default=None, help="Comma-separated CUDA ids. Example: 0,1,2,3")
    parser.add_argument("--start", type=int, default=0, help="First record index in the prompt file.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to run.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--num_output_frames", type=int, default=None, help="Override latent frame count.")
    parser.add_argument("--num_frame_per_block", type=int, default=None, help="Override latent frames per denoising block.")
    parser.add_argument("--wan_model_root", default=None, help="Root directory containing Wan2.2-TI2V-5B model files.")
    parser.add_argument("--scene_cut_prefix", default=DEFAULT_SCENE_CUT_PREFIX)
    parser.add_argument("--uniform_shot_blocks", action="store_true", help="Ignore source switch points and distribute blocks evenly across shots.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hf_repo_id", default=None, help="Optional Hugging Face repo id, e.g. Orannue/Baseline_results.")
    parser.add_argument("--hf_upload_path", default=None, help="Optional path inside the Hugging Face repo.")
    parser.add_argument("--hf_repo_type", default="dataset", choices=["dataset", "model", "space"])
    parser.add_argument("--hf_token", default=None, help="Optional token. If omitted, huggingface_hub uses the cached login/env token.")
    return parser.parse_args()


def load_prompt_records(path: str | os.PathLike) -> list[dict[str, Any]]:
    prompt_path = Path(path)
    with prompt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and "records" in data:
        records = data["records"]
    elif isinstance(data, dict):
        records = []
        for key in sorted(data.keys(), key=_natural_sort_key):
            item = dict(data[key])
            item.setdefault("idx", _idx_from_key(key, len(records)))
            if "shots" in item and "prompts" not in item:
                item["prompts"] = [shot.get("caption", "") for shot in item["shots"]]
            records.append(item)
    else:
        raise ValueError(f"Unsupported prompt file format: {prompt_path}")

    if not records:
        raise ValueError(f"No prompt records found in {prompt_path}")
    return [dict(record) for record in records]


def _idx_from_key(key: str, fallback: int) -> int:
    match = re.search(r"(\d+)$", key)
    return int(match.group(1)) if match else fallback


def _natural_sort_key(key: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", key)
    if not match:
        return key, -1
    return key[: match.start()], int(match.group(1))


def clean_shot_prompt(prompt: str) -> str:
    return FAR_SHOT_CUT_RE.sub("", str(prompt)).strip()


def cumulative_switches_to_block_counts(
    switch_latent_frames: list[int],
    source_total_latents: int,
    total_blocks: int,
) -> list[int]:
    if not switch_latent_frames:
        return [total_blocks]
    boundaries = []
    for switch in switch_latent_frames:
        boundary = int(round(float(switch) * total_blocks / float(source_total_latents)))
        boundary = max(1, min(total_blocks - 1, boundary))
        if boundaries and boundary <= boundaries[-1]:
            boundary = min(total_blocks - 1, boundaries[-1] + 1)
        boundaries.append(boundary)
    boundaries = [b for b in boundaries if 0 < b < total_blocks]
    points = [0, *boundaries, total_blocks]
    counts = [points[i + 1] - points[i] for i in range(len(points) - 1)]
    if any(count <= 0 for count in counts):
        raise ValueError(f"Invalid block counts from switches={switch_latent_frames}: {counts}")
    return counts


def normalize_block_counts(counts: list[int], total_blocks: int, num_shots: int) -> list[int]:
    if not counts:
        base = total_blocks // num_shots
        extra = total_blocks % num_shots
        return [base + (1 if i < extra else 0) for i in range(num_shots)]

    counts = [max(1, int(count)) for count in counts[:num_shots]]
    while len(counts) < num_shots:
        counts.append(1)

    delta = total_blocks - sum(counts)
    idx = 0
    while delta > 0:
        counts[idx % len(counts)] += 1
        idx += 1
        delta -= 1
    idx = len(counts) - 1
    while delta < 0:
        if counts[idx] > 1:
            counts[idx] -= 1
            delta += 1
        idx = (idx - 1) % len(counts)
    return counts


def build_block_prompts(
    record: dict[str, Any],
    total_blocks: int,
    num_frame_per_block: int,
    scene_cut_prefix: str,
    uniform_shot_blocks: bool = False,
):
    if not uniform_shot_blocks and "block_prompts" in record and len(record["block_prompts"]) == total_blocks:
        block_prompts = list(record["block_prompts"])
        block_counts = list(record.get("shot_block_counts", []))
        if not block_counts:
            block_counts = infer_counts_from_block_prompts(block_prompts, scene_cut_prefix)
        return block_prompts, block_counts

    shot_prompts = [clean_shot_prompt(prompt) for prompt in record.get("prompts", [])]
    if not shot_prompts:
        raise ValueError(f"Record idx={record.get('idx')} has no prompts")

    source = record.get("source") or record.get("source_info") or {}

    if uniform_shot_blocks:
        block_counts = normalize_block_counts([], total_blocks, len(shot_prompts))
    elif "shot_block_counts" in record:
        block_counts = normalize_block_counts(list(record["shot_block_counts"]), total_blocks, len(shot_prompts))
    elif "switch_latent_frames" in record:
        source_total = int(record.get("latent_total_frames", total_blocks * num_frame_per_block))
        block_counts = cumulative_switches_to_block_counts(list(record["switch_latent_frames"]), source_total, total_blocks)
        block_counts = normalize_block_counts(block_counts, total_blocks, len(shot_prompts))
    elif source and "switch_latent_frames" in source:
        source_total = int(source.get("latent_total_frames", total_blocks * num_frame_per_block))
        block_counts = cumulative_switches_to_block_counts(list(source["switch_latent_frames"]), source_total, total_blocks)
        block_counts = normalize_block_counts(block_counts, total_blocks, len(shot_prompts))
    else:
        block_counts = normalize_block_counts([], total_blocks, len(shot_prompts))

    block_prompts: list[str] = []
    for shot_idx, (prompt, count) in enumerate(zip(shot_prompts, block_counts)):
        for block_in_shot in range(count):
            if shot_idx > 0 and block_in_shot == 0:
                block_prompts.append(scene_cut_prefix + prompt)
            else:
                block_prompts.append(prompt)
    return block_prompts, block_counts


def infer_counts_from_block_prompts(block_prompts: list[str], scene_cut_prefix: str) -> list[int]:
    boundaries = [i for i, prompt in enumerate(block_prompts) if i > 0 and str(prompt).startswith(scene_cut_prefix)]
    points = [0, *boundaries, len(block_prompts)]
    return [points[i + 1] - points[i] for i in range(len(points) - 1)]


def shot_frame_boundaries(
    block_counts: list[int],
    num_frame_per_block: int,
    total_latent_frames: int,
    total_video_frames: int,
) -> list[int]:
    latent_boundaries = []
    running = 0
    for count in block_counts[:-1]:
        running += count * num_frame_per_block
        latent_boundaries.append(min(running, total_latent_frames))

    if total_latent_frames <= 1 or total_video_frames <= 1:
        return [min(total_video_frames, b) for b in latent_boundaries]

    ratio = float(total_video_frames - 1) / float(total_latent_frames - 1)
    boundaries = []
    for latent_idx in latent_boundaries:
        if latent_idx <= 0:
            frame_idx = 0
        elif latent_idx >= total_latent_frames:
            frame_idx = total_video_frames
        else:
            frame_idx = int(round(1 + (latent_idx - 1) * ratio))
        boundaries.append(max(1, min(total_video_frames - 1, frame_idx)))
    return boundaries


def save_sample_outputs(video: torch.Tensor, sample_dir: Path, block_counts: list[int], config, fps: int) -> dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_video(video, sample_dir / "full.mp4", fps=fps)

    nfpb = int(getattr(config, "num_frame_per_block", 1))
    total_latent_frames = int(getattr(config, "num_output_frames", config.image_or_video_shape[1]))
    total_video_frames = int(video.shape[0])
    boundaries = shot_frame_boundaries(block_counts, nfpb, total_latent_frames, total_video_frames)
    points = [0, *boundaries, total_video_frames]
    shot_frame_counts = []
    for shot_idx in range(len(points) - 1):
        start, end = points[shot_idx], points[shot_idx + 1]
        shot = video[start:end]
        shot_frame_counts.append(int(shot.shape[0]))
        save_video(shot, sample_dir / f"shot_{shot_idx + 1}.mp4", fps=fps)

    return {
        "total_video_frames": total_video_frames,
        "shot_frame_boundaries": boundaries,
        "shot_frame_counts": shot_frame_counts,
    }


def sample_outputs_complete(sample_dir: Path, expected_shots: int) -> bool:
    if not (sample_dir / "full.mp4").exists():
        return False
    if not (sample_dir / "metadata.json").exists():
        return False
    return all((sample_dir / f"shot_{shot_idx}.mp4").exists() for shot_idx in range(1, expected_shots + 1))


def configure_config(args: argparse.Namespace):
    config = normalize_config(OmegaConf.load(args.config_path))
    if args.wan_model_root is not None:
        config.wan_model_root = args.wan_model_root
    if args.num_frame_per_block is not None:
        config.num_frame_per_block = args.num_frame_per_block
        config.model_kwargs.num_frame_per_block = args.num_frame_per_block
    if args.num_output_frames is not None:
        config.num_output_frames = args.num_output_frames
        config.image_or_video_shape[1] = args.num_output_frames
    else:
        config.num_output_frames = int(getattr(config, "num_output_frames", config.image_or_video_shape[1]))
    return config


def build_pipeline(config, checkpoint_path: str, device: torch.device):
    torch.set_grad_enabled(False)
    pipe = CausalDiffusionInferencePipeline(config, device=device)
    load_generator_checkpoint(pipe.generator, checkpoint_path, use_ema=bool(getattr(config, "use_ema", False)))
    pipe = pipe.to(device=device, dtype=torch.bfloat16)
    place_vae_for_streaming(pipe, config)
    pipe.generator.model.eval().requires_grad_(False)
    return pipe


def run_worker(rank: int, world_size: int, args: argparse.Namespace, gpu_ids: list[int] | None):
    if gpu_ids is not None:
        device_id = gpu_ids[rank]
    else:
        device_id = rank
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    config = configure_config(args)
    records = load_prompt_records(args.prompts_path)
    records = records[args.start :]
    if args.limit is not None:
        records = records[: args.limit]

    total_blocks = int(config.num_output_frames) // int(config.num_frame_per_block)
    if int(config.num_output_frames) % int(config.num_frame_per_block) != 0:
        raise ValueError("num_output_frames must be divisible by num_frame_per_block")

    fps = args.fps
    if fps is None:
        fps = int(wan_default_config[getattr(config.model_kwargs, "model_name", "Wan2.2-TI2V-5B")]["fps"])

    pipe = build_pipeline(config, args.checkpoint_path, device)
    output_dir = Path(args.output_dir)

    for local_i, record in enumerate(records):
        if local_i % world_size != rank:
            continue

        idx = int(record.get("idx", record.get("index", args.start + local_i)))
        sample_dir = output_dir / f"video{idx}"

        block_prompts, block_counts = build_block_prompts(
            record,
            total_blocks=total_blocks,
            num_frame_per_block=int(config.num_frame_per_block),
            scene_cut_prefix=args.scene_cut_prefix,
            uniform_shot_blocks=args.uniform_shot_blocks,
        )

        if not args.overwrite:
            if sample_outputs_complete(sample_dir, len(block_counts)):
                print(f"[rank {rank}] skip idx={idx}: complete outputs exist")
                continue
            if (sample_dir / "full.mp4").exists():
                print(f"[rank {rank}] rerun idx={idx}: incomplete outputs found in {sample_dir}")

        generator = torch.Generator(device=device).manual_seed(args.seed + idx)
        shape = config.image_or_video_shape
        noise = torch.randn(
            [1, int(config.num_output_frames), int(shape[2]), int(shape[3]), int(shape[4])],
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )

        print(f"[rank {rank}] generating idx={idx}, blocks={block_counts}")
        video = pipe.inference(noise=noise, text_prompts=[block_prompts])[0]
        meta = save_sample_outputs(video, sample_dir, block_counts, config, fps)
        meta.update(
            {
                "idx": idx,
                "seed": args.seed + idx,
                "fps": fps,
                "shot_block_counts": block_counts,
                "block_prompts": block_prompts,
                "source_summary": record.get("random_concept_summary") or record.get("global_caption"),
            }
        )
        with (sample_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if hasattr(pipe, "clear_cache"):
            pipe.clear_cache()
        if hasattr(pipe.vae, "model") and hasattr(pipe.vae.model, "clear_cache"):
            pipe.vae.model.clear_cache()
        torch.cuda.empty_cache()


def launch_from_torchrun(args: argparse.Namespace) -> bool:
    if "LOCAL_RANK" not in os.environ:
        return False
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    run_worker(local_rank, world_size, args, None)
    if dist.is_initialized():
        dist.barrier()
    if local_rank == 0:
        upload_to_huggingface(args)
    if dist.is_initialized():
        dist.destroy_process_group()
    return True


def upload_to_huggingface(args: argparse.Namespace) -> None:
    if not args.hf_repo_id:
        return

    folder_path = Path(args.output_dir)
    if not folder_path.exists():
        raise FileNotFoundError(f"Output directory does not exist: {folder_path}")

    path_in_repo = args.hf_upload_path
    if path_in_repo is not None:
        path_in_repo = path_in_repo.strip().strip("/")
        if path_in_repo == "":
            path_in_repo = None

    cmd = [
        "hf",
        "upload",
        args.hf_repo_id,
        str(folder_path),
        path_in_repo or "",
        "--repo-type",
        args.hf_repo_type,
    ]
    if args.hf_token:
        cmd.extend(["--token", args.hf_token])

    printable = " ".join(cmd[:-1] + ["***"] if args.hf_token else cmd)
    print(f"[hf] running: {printable}")
    subprocess.run(cmd, check=True)
    print("[hf] upload complete")


def main():
    args = parse_args()
    if launch_from_torchrun(args):
        return

    if args.gpus:
        gpu_ids = [int(part.strip()) for part in args.gpus.split(",") if part.strip()]
    else:
        gpu_ids = [0]

    if len(gpu_ids) == 1:
        run_worker(0, 1, args, gpu_ids)
    else:
        mp.spawn(run_worker, args=(len(gpu_ids), args, gpu_ids), nprocs=len(gpu_ids), join=True)
    upload_to_huggingface(args)


if __name__ == "__main__":
    main()
