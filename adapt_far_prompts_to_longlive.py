import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_SCENE_CUT_PREFIX = "The scene transitions. "
SHOT_CUT_RE = re.compile(r"^\s*\[shot cut\]\s*")
CHARACTER_TAG_RE = re.compile(r"\[character(\d+)\]")
ACTION_START_RE = re.compile(
    r"\b("
    r"adjusting|arriving|blowing|carrying|checking|clapping|cooking|dancing|doing|"
    r"ducking|entering|finishing|gathering|gesturing|holding|hoisting|jogging|"
    r"kneeling|lifting|looking|moving|noticing|opening|pausing|performing|placing|"
    r"playing|pointing|presenting|pulling|reaching|rolling|running|settling|"
    r"sipping|sliding|smiling|spinning|standing|stepping|stopping|studying|swiping|"
    r"turning|unfolding|walking|wiping|whispering"
    r")\b",
    re.IGNORECASE,
)

UNWANTED_CLAUSE_STARTS = (
    "positioned",
    "perspective",
    "mandatory instruction",
    "with mandatory instruction",
    "keeping ",
    "keep ",
    "gentle handheld",
    "medium close-up shot",
    "simple tripod",
    "simple tripod-mounted",
    "steady shoulder-level",
    "wide shot",
    "medium shot",
    "close-up shot",
    "low-angle shot",
    "over-the-shoulder shot",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt FAR multi-shot prompts to LongLive block-level prompts.")
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--output_path", default="eval_caption_multishot_t2v_100_longlive.json")
    parser.add_argument("--num_output_frames", type=int, default=192)
    parser.add_argument("--num_frame_per_block", type=int, default=8)
    parser.add_argument("--scene_cut_prefix", default=DEFAULT_SCENE_CUT_PREFIX)
    parser.add_argument("--uniform_shot_blocks", action="store_true")
    return parser.parse_args()


def clean_raw_prompt(prompt: str) -> str:
    text = SHOT_CUT_RE.sub("", str(prompt)).strip()
    return re.sub(r"\s+", " ", text).strip()


def infer_pronouns(text: str) -> tuple[str, str]:
    lower = text.lower()
    if any(word in lower for word in ("woman", "female", "girl", "lady", "mother", "sister", "actress")):
        return "She", "her"
    if any(word in lower for word in ("man", "male", "boy", "gentleman", "father", "brother", "bearded")):
        return "He", "him"
    if any(word in lower for word in ("hands", "people", "couple", "group", "dancers", "children")):
        return "They", "them"
    return "The same subject", "them"


def normalize_subject(subject: str) -> str:
    subject = subject.strip(" ,")
    subject = re.sub(r"\s+", " ", subject)
    if not subject:
        return "The main subject"
    if subject[0].islower():
        subject = subject[0].upper() + subject[1:]
    return subject


def is_unwanted_clause(clause: str) -> bool:
    lower = clause.strip().lower()
    return any(lower.startswith(prefix) for prefix in UNWANTED_CLAUSE_STARTS)


def trim_clause_to_action(clause: str) -> str:
    clause = clause.strip()
    match = ACTION_START_RE.search(clause)
    if not match:
        return clause
    lead = clause[: match.start()].lower()
    if lead.startswith("with ") or "hair" in lead or "wearing" in lead:
        return clause[match.start():].strip()
    return clause


def split_first_camera_sentence(text: str) -> tuple[str, str]:
    sentences = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    first = sentences[0].strip()
    rest = sentences[1].strip() if len(sentences) > 1 else ""
    if rest.startswith("The camera"):
        return first, rest
    return first, ""


def extract_subject_and_action(raw_prompt: str) -> tuple[str, str, str]:
    """Return (subject_once, action_context, camera_sentence)."""
    text = clean_raw_prompt(raw_prompt)
    text = CHARACTER_TAG_RE.sub(lambda m: "__CHAR1__" if m.group(1) == "1" else f"character {m.group(1)}", text)
    first_sentence, camera_sentence = split_first_camera_sentence(text)
    clauses = [clause.strip() for clause in first_sentence.split(",") if clause.strip()]

    tag_index = next((i for i, clause in enumerate(clauses) if clause == "__CHAR1__"), -1)
    search_start = tag_index + 1 if tag_index >= 0 else 0
    action_index = next(
        (i for i in range(search_start, len(clauses)) if ACTION_START_RE.search(clauses[i])),
        search_start,
    )

    subject_clauses = clauses[search_start:action_index]
    action_clauses = []
    for clause in clauses[action_index:]:
        if is_unwanted_clause(clause):
            continue
        clause = clause.replace("__CHAR1__", "").strip()
        if not action_clauses:
            clause = trim_clause_to_action(clause)
        action_clauses.append(clause)

    subject = normalize_subject(", ".join(subject_clauses))
    action_context = ", ".join(action_clauses).strip(" ,")
    action_context = CHARACTER_TAG_RE.sub(lambda m: "the other character", action_context)
    action_context = action_context.replace("__CHAR1__", "").strip(" ,")
    return subject, action_context, camera_sentence


def as_continuity_prompt(raw_prompt: str, shot_idx: int, first_subject: str, pronoun: str) -> str:
    subject, action_context, camera_sentence = extract_subject_and_action(raw_prompt)
    active_subject = subject if shot_idx == 0 else pronoun
    if shot_idx == 0 and ", with " in active_subject and not active_subject.endswith(","):
        active_subject += ","
    if action_context:
        prompt = f"{active_subject} is {action_context}"
    else:
        prompt = clean_raw_prompt(raw_prompt)
        prompt = CHARACTER_TAG_RE.sub(active_subject, prompt)
    if camera_sentence:
        prompt = f"{prompt}. {camera_sentence}"
    if not prompt.endswith("."):
        prompt += "."
    return re.sub(r"\s+", " ", prompt).strip()


def adapt_record_prompts(raw_prompts: list[str]) -> list[str]:
    first_subject, _, _ = extract_subject_and_action(raw_prompts[0])
    pronoun, _ = infer_pronouns(first_subject + " " + " ".join(raw_prompts))
    return [
        as_continuity_prompt(prompt, shot_idx, first_subject, pronoun)
        for shot_idx, prompt in enumerate(raw_prompts)
    ]


def block_counts_from_switches(
    switches: list[int],
    source_total_latents: int,
    total_blocks: int,
    num_shots: int,
) -> list[int]:
    if not switches:
        return uniform_counts(total_blocks, num_shots)

    boundaries: list[int] = []
    for switch in switches:
        boundary = round(float(switch) * total_blocks / float(source_total_latents))
        boundary = max(1, min(total_blocks - 1, int(boundary)))
        if boundaries and boundary <= boundaries[-1]:
            boundary = min(total_blocks - 1, boundaries[-1] + 1)
        if 0 < boundary < total_blocks:
            boundaries.append(boundary)

    points = [0, *boundaries, total_blocks]
    counts = [points[i + 1] - points[i] for i in range(len(points) - 1)]
    return normalize_counts(counts, total_blocks, num_shots)


def uniform_counts(total_blocks: int, num_shots: int) -> list[int]:
    base, extra = divmod(total_blocks, num_shots)
    return [base + (1 if i < extra else 0) for i in range(num_shots)]


def normalize_counts(counts: list[int], total_blocks: int, num_shots: int) -> list[int]:
    if not counts:
        return uniform_counts(total_blocks, num_shots)

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


def build_block_prompts(prompts: list[str], counts: list[int], scene_cut_prefix: str) -> list[str]:
    block_prompts: list[str] = []
    for shot_idx, (prompt, count) in enumerate(zip(prompts, counts)):
        for block_idx in range(count):
            if shot_idx > 0 and block_idx == 0:
                block_prompts.append(scene_cut_prefix + prompt)
            else:
                block_prompts.append(prompt)
    return block_prompts


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, dict) and "records" in data:
        return [dict(item) for item in data["records"]]
    raise ValueError(f"Unsupported FAR prompt format: {path}")


def main() -> None:
    args = parse_args()
    if args.num_output_frames % args.num_frame_per_block != 0:
        raise ValueError("num_output_frames must be divisible by num_frame_per_block")

    total_blocks = args.num_output_frames // args.num_frame_per_block
    records = []
    for fallback_idx, item in enumerate(load_records(Path(args.source_path)), start=1):
        raw_prompts = list(item.get("prompts", []))
        prompts = adapt_record_prompts(raw_prompts)
        if not prompts:
            raise ValueError(f"Record {fallback_idx} has no prompts")

        source_total = int(item.get("latent_total_frames", args.num_output_frames))
        switches = [int(value) for value in item.get("switch_latent_frames", [])]
        if args.uniform_shot_blocks:
            counts = uniform_counts(total_blocks, len(prompts))
        else:
            counts = block_counts_from_switches(switches, source_total, total_blocks, len(prompts))

        idx = int(item.get("idx", item.get("index", fallback_idx)))
        records.append(
            {
                "idx": idx,
                "random_concept_summary": item.get("random_concept_summary", ""),
                "prompts": prompts,
                "shots": [
                    {"shot": shot_idx + 1, "prompt": prompt}
                    for shot_idx, prompt in enumerate(prompts)
                ],
                "shot_block_counts": counts,
                "block_prompts": build_block_prompts(prompts, counts, args.scene_cut_prefix),
                "latent_total_frames": args.num_output_frames,
                "num_frame_per_block": args.num_frame_per_block,
                "source_latent_total_frames": source_total,
                "source_switch_latent_frames": switches,
                "prompt_style": "natural_continuity",
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
