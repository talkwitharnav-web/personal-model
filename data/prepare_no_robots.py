"""No Robots - human-written instruction following, for the v2 mix.

Section 10: SODA and No Robots go INTO the combined mix, not a third stage.

10k human-written prompt/response pairs across categories (Generation, Open QA, Brainstorm,
Summarize, Rewrite, Coding, Classify, Closed QA, Chat, Extract). Small but high quality, and the
instructions are far more varied than the templated datasets already in the mix - which is the
same reason the yes/no set needed 53 phrasings rather than 3.

Schema verified live before writing this (section 6):
    prompt    str                            - the instruction
    messages  list[{content, role}]          - the full exchange, roles 'user'/'assistant'
    category  str

Uses `messages` rather than `prompt`, because some rows are multi-turn and the prompt field only
holds the opening instruction.
"""
import sys
from pathlib import Path

from datasets import load_dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path("data/no_robots")
MAX_CHARS = 2_400          # keeps examples inside the 768-token window after tokenizing


def build(split_name, hf_split):
    dataset = load_dataset("HuggingFaceH4/no_robots", split=hf_split)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    categories = {}

    with open(OUT_DIR / f"{split_name}.txt", "w", encoding="utf-8") as handle:
        for row in dataset:
            messages = row.get("messages") or []
            turns = []
            for message in messages:
                role = str(message.get("role", "")).lower()
                content = " ".join(str(message.get("content", "")).split())
                if not content:
                    continue
                if role == "user":
                    turns.append(("User", content))
                elif role == "assistant":
                    turns.append(("Assistant", content))
                # 'system' rows are dropped: nothing else in the mix has a system prompt, and
                # introducing one format the model sees nowhere else teaches a spurious cue.

            # Must end on an Assistant turn - a trailing user turn teaches the model to stop
            # before answering (the PersonaChat bug, section 7).
            while turns and turns[-1][0] != "Assistant":
                turns.pop()
            if len(turns) < 2 or not any(role == "User" for role, _ in turns):
                skipped += 1
                continue

            body = "\n".join(f"{role}: {text}" for role, text in turns)
            if len(body) > MAX_CHARS:
                skipped += 1
                continue

            handle.write(f"{body}\n<|endoftext|>\n")
            written += 1
            category = row.get("category", "?")
            categories[category] = categories.get(category, 0) + 1

    print(f"  {split_name}: {written:,} written, {skipped:,} skipped")
    if categories:
        spread = ", ".join(f"{k} {v}" for k, v in
                           sorted(categories.items(), key=lambda kv: -kv[1])[:10])
        print(f"    categories: {spread}")
    return written


def main():
    print("preparing No Robots")
    build("train", "train")
    build("validation", "test")

    sample = (OUT_DIR / "train.txt").read_text(encoding="utf-8")
    print("\nfirst example:")
    print("  " + sample.split("<|endoftext|>")[0].strip().replace("\n", "\n  ")[:600])
    if "\nAssistant:" not in sample[:20_000]:
        raise SystemExit("FAIL: no \\nAssistant: marker - every tokenize script would discard this")
    print("\nverified: \\nAssistant: marker present")


if __name__ == "__main__":
    main()
