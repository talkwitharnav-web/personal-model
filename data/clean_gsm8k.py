"""Strip GSM8K's '#### <answer>' artifact so it can join the v2 fine-tune mix.

Section 10 reverses the v1 exclusion of GSM8K: the objection was Dark's, and it was about
disrupting a nearly-finished training run - "training now would look weird in its training
timeline" - which cannot apply to a model trained from zero. So GSM8K belongs in the v2 mix.

But the shipped file is not usable as-is. Every one of the 7,473 answers ends with GSM8K's
machine-readable marker:

    Natalia sold 48/2 = 24 clips in May.
    Natalia sold 48+24 = 72 clips altogether in April and May.
    #### 72

'####' is a dataset convention for graders, not English. Training on it teaches the model to emit
'####' before every final answer - a formatting artifact leaking straight into behaviour, which is
the same class of problem as the untrained stop token in section 7. The <<48/2=24>> calculator
annotations were already stripped by prepare_gsm8k.py; this one was missed.

Rewrites the marker as a plain sentence, keeping the number, so the reasoning chain still ends on
a clearly-marked final answer.

Writes to data/gsm8k_clean/ so the original stays intact for comparison.
"""
import random
import re
import sys
from pathlib import Path

from tools.audit_finetune_datasets import parse_record

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SOURCE_DIR = Path("data/gsm8k")
OUT_DIR = Path("data/gsm8k_clean")

FINAL = re.compile(r"\n?####\s*([\-\d,.]+)\s*$")
# Varied so the model learns "state the final answer", not one memorised string - the same
# lesson as the yes/no template fix, where three templates produced a model that only handled
# three phrasings. Five was still thin; a user asking "what's 3 x 14" should get an answer
# whether or not the model happens to reach for the one phrasing it was drilled on.
ENDINGS = [
    "The answer is {n}.",
    "So the answer is {n}.",
    "The final answer is {n}.",
    "That gives {n}.",
    "So it is {n}.",
    "This means the answer is {n}.",
    "So she has {n}.",
    "The total is {n}.",
    "That comes to {n}.",
    "Therefore the answer is {n}.",
    "Which gives us {n}.",
    "So the total comes to {n}.",
    "The result is {n}.",
    "In total, {n}.",
    "So we get {n}.",
    "That works out to {n}.",
    "Altogether that is {n}.",
    "So the answer must be {n}.",
]
# A few endings only make sense for a running total rather than a person, so the generic ones
# are weighted up and the specific ones stay rare.
ENDING_WEIGHTS = [6, 5, 4, 3, 3, 3, 1, 3, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2]


def clean_answer(answer, rng):
    match = FINAL.search(answer)
    if not match:
        return None
    body = answer[:match.start()].rstrip()
    number = match.group(1).strip()
    ending = rng.choices(ENDINGS, weights=ENDING_WEIGHTS)[0].format(n=number)
    return f"{body}\n{ending}" if body else ending


def convert(split_name, rng):
    source = SOURCE_DIR / f"{split_name}.txt"
    if not source.exists():
        print(f"  missing {source}, skipping")
        return 0

    text = source.read_text(encoding="utf-8", errors="replace")
    parts = [p for p in text.split("<|endoftext|>") if p.strip()]
    del text

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with open(OUT_DIR / f"{split_name}.txt", "w", encoding="utf-8") as handle:
        for part in parts:
            record = parse_record(part)
            if not record:
                skipped += 1
                continue
            answer = clean_answer(record["answer"], rng)
            if answer is None:
                skipped += 1
                continue
            handle.write(f"User: {record['question']}\n"
                         f"Assistant: {answer}\n<|endoftext|>\n")
            written += 1

    print(f"  {split_name}: {written:,} written, {skipped:,} skipped")
    return written


def main():
    rng = random.Random(42)
    print("stripping GSM8K's #### marker")
    for split_name in ("train", "validation"):
        convert(split_name, rng)

    sample = (OUT_DIR / "train.txt").read_text(encoding="utf-8")
    print("\nfirst two cleaned examples:")
    for part in sample.split("<|endoftext|>")[:2]:
        if part.strip():
            print("  " + part.strip().replace("\n", "\n  ")[:420])
            print()
    if "####" in sample:
        raise SystemExit("FAIL: '####' still present in the output")
    print("verified: no '####' remains in the cleaned file")


if __name__ == "__main__":
    main()
