import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SCENE_CUT_PREFIX = "The scene transitions. "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MSAVBench generation inputs to LongLive multi-shot records."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--group_output_dir", default=None)
    parser.add_argument("--num_frame_per_block", type=int, default=8)
    parser.add_argument("--temporal_compression_ratio", type=int, default=4)
    parser.add_argument("--scene_cut_prefix", default=DEFAULT_SCENE_CUT_PREFIX)
    return parser.parse_args()


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def video_frames_to_latent_frames(frames: int, temporal_compression_ratio: int) -> int:
    if frames <= 0:
        return 1
    return max(1, round_half_up(((frames - 1) / float(temporal_compression_ratio)) + 1))


def latent_frames_to_blocks(latent_frames: int, num_frame_per_block: int) -> int:
    return max(1, round_half_up(latent_frames / float(num_frame_per_block)))


def build_block_prompts(prompts: list[str], counts: list[int], scene_cut_prefix: str) -> list[str]:
    block_prompts: list[str] = []
    for shot_idx, (prompt, count) in enumerate(zip(prompts, counts)):
        for block_idx in range(count):
            if shot_idx > 0 and block_idx == 0 and scene_cut_prefix:
                block_prompts.append(scene_cut_prefix + prompt)
            else:
                block_prompts.append(prompt)
    return block_prompts


def convert_record(
    record: dict[str, Any],
    fallback_idx: int,
    num_frame_per_block: int,
    temporal_compression_ratio: int,
    scene_cut_prefix: str,
) -> dict[str, Any]:
    prompts = [str(prompt) for prompt in record.get("prompts", [])]
    frames_per_shot = [int(frames) for frames in record.get("frames_per_shot", [])]
    if not prompts:
        raise ValueError(f"Record {fallback_idx} has no prompts")
    if len(prompts) != len(frames_per_shot):
        raise ValueError(
            f"Record {fallback_idx} has {len(prompts)} prompts but "
            f"{len(frames_per_shot)} frame counts"
        )

    latent_frames_per_shot = [
        video_frames_to_latent_frames(frames, temporal_compression_ratio)
        for frames in frames_per_shot
    ]
    shot_block_counts = [
        latent_frames_to_blocks(latent_frames, num_frame_per_block)
        for latent_frames in latent_frames_per_shot
    ]

    sample_id = str(record.get("sample_id", record.get("idx", fallback_idx)))
    try:
        idx = int(sample_id)
    except ValueError:
        idx = fallback_idx

    num_output_frames = sum(shot_block_counts) * num_frame_per_block
    converted = {
        "idx": idx,
        "sample_id": sample_id,
        "prompts": prompts,
        "shots": [
            {
                "shot": shot_idx + 1,
                "prompt": prompt,
                "source_video_frames": frames_per_shot[shot_idx],
                "latent_frames": latent_frames_per_shot[shot_idx],
                "blocks": shot_block_counts[shot_idx],
            }
            for shot_idx, prompt in enumerate(prompts)
        ],
        "frames_per_shot": frames_per_shot,
        "latent_frames_per_shot": latent_frames_per_shot,
        "shot_block_counts": shot_block_counts,
        "block_prompts": build_block_prompts(prompts, shot_block_counts, scene_cut_prefix),
        "num_output_frames": num_output_frames,
        "num_frame_per_block": num_frame_per_block,
        "temporal_compression_ratio": temporal_compression_ratio,
        "source_total_video_frames": sum(frames_per_shot),
        "source_total_latent_frames": sum(latent_frames_per_shot),
        "block_count_method": "round((round((frames - 1) / temporal_compression_ratio + 1)) / num_frame_per_block), min 1",
    }
    return converted


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {input_path}")

    converted = [
        convert_record(
            record,
            fallback_idx=i,
            num_frame_per_block=args.num_frame_per_block,
            temporal_compression_ratio=args.temporal_compression_ratio,
            scene_cut_prefix=args.scene_cut_prefix,
        )
        for i, record in enumerate(records)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    by_num_output_frames: dict[int, int] = {}
    for record in converted:
        num_output_frames = int(record["num_output_frames"])
        by_num_output_frames[num_output_frames] = by_num_output_frames.get(num_output_frames, 0) + 1

    if args.group_output_dir:
        group_output_dir = Path(args.group_output_dir)
        group_output_dir.mkdir(parents=True, exist_ok=True)
        for num_output_frames in sorted(by_num_output_frames):
            group_records = [
                record
                for record in converted
                if int(record["num_output_frames"]) == num_output_frames
            ]
            group_path = group_output_dir / f"msavbench_num_output_frames_{num_output_frames}.json"
            with group_path.open("w", encoding="utf-8") as f:
                json.dump(group_records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(converted)} records to {output_path}")
    print("num_output_frames distribution:")
    for num_output_frames, count in sorted(by_num_output_frames.items()):
        print(f"  {num_output_frames}: {count}")


if __name__ == "__main__":
    main()
