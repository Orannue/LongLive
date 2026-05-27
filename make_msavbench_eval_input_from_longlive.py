import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy MSAVBench evaluation template and attach LongLive output paths/timing metadata."
    )
    parser.add_argument("--template_path", required=True)
    parser.add_argument("--longlive_records_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--video_root",
        default=r"C:\Users\Administrator\Desktop\LongLive\videos\msavbench_video_only",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--temporal_compression_ratio", type=int, default=4)
    return parser.parse_args()


def decoded_video_frames(latent_frames: int, temporal_compression_ratio: int) -> int:
    if latent_frames <= 0:
        return 0
    return (latent_frames - 1) * temporal_compression_ratio + 1


def shot_frame_boundaries(
    block_counts: list[int],
    num_frame_per_block: int,
    total_latent_frames: int,
    total_video_frames: int,
) -> list[int]:
    latent_boundaries: list[int] = []
    running = 0
    for count in block_counts[:-1]:
        running += count * num_frame_per_block
        latent_boundaries.append(min(running, total_latent_frames))

    if total_latent_frames <= 1 or total_video_frames <= 1:
        return [min(total_video_frames, boundary) for boundary in latent_boundaries]

    ratio = float(total_video_frames - 1) / float(total_latent_frames - 1)
    boundaries: list[int] = []
    for latent_idx in latent_boundaries:
        if latent_idx <= 0:
            frame_idx = 0
        elif latent_idx >= total_latent_frames:
            frame_idx = total_video_frames
        else:
            frame_idx = int(round(1 + (latent_idx - 1) * ratio))
        boundaries.append(max(1, min(total_video_frames - 1, frame_idx)))
    return boundaries


def expected_shot_frame_counts(record: dict[str, Any], temporal_compression_ratio: int) -> list[int]:
    block_counts = [int(value) for value in record["shot_block_counts"]]
    num_frame_per_block = int(record["num_frame_per_block"])
    total_latent_frames = int(record["num_output_frames"])
    total_video_frames = decoded_video_frames(total_latent_frames, temporal_compression_ratio)
    boundaries = shot_frame_boundaries(
        block_counts,
        num_frame_per_block,
        total_latent_frames,
        total_video_frames,
    )
    points = [0, *boundaries, total_video_frames]
    return [points[i + 1] - points[i] for i in range(len(points) - 1)]


def main() -> None:
    args = parse_args()
    template_path = Path(args.template_path)
    records_path = Path(args.longlive_records_path)
    output_path = Path(args.output_path)

    with template_path.open("r", encoding="utf-8") as f:
        eval_items = json.load(f)
    with records_path.open("r", encoding="utf-8") as f:
        longlive_records = json.load(f)

    if not isinstance(eval_items, list):
        raise ValueError(f"Expected list in {template_path}")
    if not isinstance(longlive_records, list):
        raise ValueError(f"Expected list in {records_path}")

    record_by_sample_id = {
        str(record["sample_id"]): record
        for record in longlive_records
    }
    video_root = Path(args.video_root)

    converted_items: list[dict[str, Any]] = []
    for item in eval_items:
        sample_id = str(item.get("sample_id", ""))
        if sample_id not in record_by_sample_id:
            raise ValueError(f"No LongLive record found for sample_id={sample_id}")
        record = record_by_sample_id[sample_id]

        updated = dict(item)
        generation_shot_captions = [str(prompt) for prompt in record["prompts"]]
        if len(generation_shot_captions) != len(record["shot_block_counts"]):
            raise ValueError(
                f"sample_id={sample_id} has {len(generation_shot_captions)} generation captions but "
                f"{len(record['shot_block_counts'])} LongLive shot counts"
            )

        shot_frames = expected_shot_frame_counts(record, args.temporal_compression_ratio)
        shot_durations = [round(frames / args.fps, 4) for frames in shot_frames]

        updated["video_path"] = str(video_root / f"video{sample_id}" / "full.mp4")
        updated["shot_captions"] = generation_shot_captions
        updated["template_shot_captions"] = list(item.get("shot_captions", []))
        updated["source_frames_per_shot"] = record["frames_per_shot"]
        updated["source_latent_frames_per_shot"] = record["latent_frames_per_shot"]
        updated["shot_block_counts"] = record["shot_block_counts"]
        updated["num_output_frames"] = record["num_output_frames"]
        updated["num_frame_per_block"] = record["num_frame_per_block"]
        updated["expected_shot_frame_counts"] = shot_frames
        updated["expected_shot_durations_seconds"] = shot_durations
        updated["expected_shot_timing_captions"] = [
            f"Shot {idx + 1} duration {shot_durations[idx]:.2f} seconds. {caption}"
            for idx, caption in enumerate(generation_shot_captions)
        ]
        updated["expected_total_duration_seconds"] = round(sum(shot_frames) / args.fps, 4)
        updated["longlive_output_dir"] = str(video_root / f"video{sample_id}")
        converted_items.append(updated)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted_items, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(converted_items)} evaluation records to {output_path}")


if __name__ == "__main__":
    main()
