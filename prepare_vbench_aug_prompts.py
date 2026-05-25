#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def one_line(value: object, key: str, index: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"record {index}: {key} must be a string")
    text = value.replace("\r", " ").replace("\n", " ").strip()
    if not text:
        raise ValueError(f"record {index}: {key} is empty")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="VBench JSON file")
    parser.add_argument("--prompt-out", required=True, help="Output prompt_en text file")
    parser.add_argument("--aug-out", required=True, help="Output aug_prompt_en text file")
    args = parser.parse_args()

    json_path = Path(args.json)
    with json_path.open(encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise TypeError(f"{json_path} must contain a JSON list")

    prompts = []
    aug_prompts = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"record {index}: expected an object")
        prompts.append(one_line(record.get("prompt_en"), "prompt_en", index))
        aug_prompts.append(one_line(record.get("aug_prompt_en"), "aug_prompt_en", index))

    prompt_out = Path(args.prompt_out)
    aug_out = Path(args.aug_out)
    prompt_out.parent.mkdir(parents=True, exist_ok=True)
    aug_out.parent.mkdir(parents=True, exist_ok=True)
    prompt_out.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    aug_out.write_text("\n".join(aug_prompts) + "\n", encoding="utf-8")

    print(f"Wrote {len(prompts)} prompt_en lines to {prompt_out}")
    print(f"Wrote {len(aug_prompts)} aug_prompt_en lines to {aug_out}")


if __name__ == "__main__":
    main()
