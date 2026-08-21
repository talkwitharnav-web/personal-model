"""SODA - social dialogue, for conversational register in the v2 mix.

Section 10: SODA and No Robots go INTO the combined mix, not a third stage. A third stage
reintroduces catastrophic forgetting; separate stages are only justified when the objective
genuinely differs (raw text vs instruction). The SQuAD v2 collapse was a narrow stage on top.

Honest expectation, also from section 10: 80M can learn conversational REGISTER, not a stable
persona or real opinions.

Schema verified live before writing this (section 6 - PersonaChat's column turned out to be
"Best Generated Conversation", not "Conversation"):
    narrative  str        - what is going on, used as the scene
    dialogue   list[str]  - alternating utterances
    speakers   list[str]  - who says each one

⚠ TRAILING USER TURNS. PersonaChat trained on examples ending mid-exchange in 42.8% of cases,
which teaches the model to stop before answering (section 7). SODA dialogues can end on either
speaker, so any trailing non-assistant turn is dropped here too.

Streamed, never downloaded whole: SODA is ~1.5M dialogues and this machine is simultaneously
running a 30-hour training job.
"""
import sys
from pathlib import Path

from datasets import load_dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path("data/soda")
TARGET = {"train": 40_000, "validation": 1_000}
MIN_TURNS = 4
MAX_TURNS = 12
MAX_CHARS = 2_400          # keeps examples inside the 768-token window after tokenizing


def build(split_name, hf_split, target):
    dataset = load_dataset("allenai/soda", split=hf_split, streaming=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = skipped = 0

    with open(OUT_DIR / f"{split_name}.txt", "w", encoding="utf-8") as handle:
        for row in dataset:
            if written >= target:
                break
            dialogue = row.get("dialogue") or []
            speakers = row.get("speakers") or []
            if len(dialogue) < MIN_TURNS or len(dialogue) != len(speakers):
                skipped += 1
                continue

            dialogue = dialogue[:MAX_TURNS]
            speakers = speakers[:MAX_TURNS]

            # Two speakers alternating: the first is the user, the second the assistant.
            first = speakers[0]
            turns = []
            for speaker, utterance in zip(speakers, dialogue):
                utterance = " ".join(str(utterance).split())
                if not utterance:
                    continue
                role = "User" if speaker == first else "Assistant"
                turns.append((role, utterance))

            # Must end on an Assistant turn, or the example teaches the model to stop before
            # answering - the PersonaChat bug.
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
            if written % 10_000 == 0:
                print(f"    {written:,} written")

    print(f"  {split_name}: {written:,} written, {skipped:,} skipped")
    return written


def main():
    print("preparing SODA (streamed)")
    build("train", "train", TARGET["train"])
    build("validation", "validation", TARGET["validation"])

    sample = (OUT_DIR / "train.txt").read_text(encoding="utf-8")
    print("\nfirst example:")
    print("  " + sample.split("<|endoftext|>")[0].strip().replace("\n", "\n  ")[:600])
    if "\nAssistant:" not in sample[:20_000]:
        raise SystemExit("FAIL: no \\nAssistant: marker - every tokenize script would discard this")
    print("\nverified: \\nAssistant: marker present")


if __name__ == "__main__":
    main()
