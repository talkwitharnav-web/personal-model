"""Check the typo augmentation did what it claims, before anyone trains on it.

Two things must hold, and both are easy to get wrong:

1. ASSISTANT TURNS MUST BE UNTOUCHED. Packing masks everything except the assistant span, so the
   answer is the only text carrying gradient. Corrupting it would train the model to *write* badly
   - the exact opposite of the goal - and it would be invisible in any summary statistic.
2. The corruption has to be REAL. If almost every "messy" copy differs only by lowercasing, the
   set teaches capitalisation robustness and nothing about typos, while looking like a fix.

This compares the augmented file against the sources it was built from.
"""
import random
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SOURCES = ["no_robots", "oasst1", "soda", "personachat", "definitions",
           "yesno", "squad", "sciq", "boolq", "gsm8k_clean"]


def blocks_of(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return [b.strip() for b in text.split("<|endoftext|>") if b.strip()]


def assistant_spans(block):
    return [line for line in block.split("\n") if line.startswith("Assistant:")]


def context_spans(block):
    return [line for line in block.split("\n") if line.startswith("Context:")]


def first_user(block):
    for line in block.split("\n"):
        if line.startswith("User:"):
            return line[len("User:"):].strip()
    return ""


def main():
    aug = blocks_of("data/typo_aug/train.txt")
    print(f"augmented blocks: {len(aug):,}\n")

    # ⚠ Sample ACROSS the file, not off the front. The augmenter writes dataset by dataset, so the
    # first N blocks are all chat data - an earlier version of this check read the head and
    # reported "context lines checked: 0", cheerfully passing a test it had never run.
    rng = random.Random(3)
    probe = rng.sample(aug, min(8000, len(aug)))

    # Every assistant line in the augmented file must exist verbatim in some source file.
    originals = set()
    original_contexts = set()
    original_users = set()
    for name in SOURCES:
        path = Path("data") / name / "train.txt"
        if not path.exists():
            continue
        for block in blocks_of(path):
            for line in assistant_spans(block):
                originals.add(line)
            for line in context_spans(block):
                original_contexts.add(line)
            user = first_user(block)
            if user:
                original_users.add(user)
    print(f"distinct assistant lines across sources: {len(originals):,}")

    checked = corrupted_answers = 0
    ctx_checked = corrupted_contexts = 0
    for block in probe:
        for line in assistant_spans(block):
            checked += 1
            if line not in originals:
                corrupted_answers += 1
        for line in context_spans(block):
            ctx_checked += 1
            if line not in original_contexts:
                corrupted_contexts += 1
    print(f"assistant lines checked: {checked:,} | NOT found in the originals: {corrupted_answers:,}")
    print("  -> " + ("PASS - answers are untouched" if corrupted_answers == 0
                     else "FAIL - the augmenter is corrupting training targets"))
    print(f"context lines checked:   {ctx_checked:,} | NOT found in the originals: "
          f"{corrupted_contexts:,}")
    print("  -> " + ("PASS - retrieved passages are untouched (a real user types badly, the "
                     "retriever does not)" if corrupted_contexts == 0
                     else "FAIL - the augmenter is corrupting retrieved context"))

    # The whole point is that the user turn CHANGED. A messy copy identical to its original is a
    # duplicate row teaching nothing, and would quietly inflate the dataset.
    unchanged = sum(1 for block in probe if first_user(block) in original_users)
    print(f"\nuser turns identical to an original: {unchanged:,} of {len(probe):,} "
          f"({100*unchanged/len(probe):.1f}%)")
    print("  -> " + ("PASS - the copies really are different" if unchanged < 0.1 * len(probe)
                     else "FAIL - too many copies are byte-identical duplicates"))

    # How much of the change is more than lowercasing?
    sample = probe
    lower_only = typo_like = shorthand_like = 0
    shorthand = re.compile(r"\b(u|ur|pls|thx|ty|idk|tbh|imo|btw|asap|omg|lmao|lol|bc|b4|ppl|"
                           r"rly|prob|def|abt|smth|tmrw|wat|tho|w/|w/o)\b")
    for block in sample:
        user = first_user(block)
        if not user:
            continue
        if shorthand.search(user):
            shorthand_like += 1
        # a word with a letter pattern no English dictionary would produce is a decent proxy
        if re.search(r"\b\w*([bcdfghjklmnpqrstvwxz])\1{2,}\w*\b", user) or \
           re.search(r"\b[a-z]*[qwxz][qwxz][a-z]*\b", user):
            typo_like += 1
        if user == user.lower() and not shorthand.search(user):
            lower_only += 1

    n = len(sample)
    print(f"\nof {n:,} sampled messy user turns:")
    print(f"  contain shorthand (u/pls/idk/...): {shorthand_like:,} ({100*shorthand_like/n:.0f}%)")
    print(f"  all lowercase:                     {lower_only:,} ({100*lower_only/n:.0f}%)")

    print("\n--- 8 messy user turns, read them ---")
    rng = random.Random(3)
    for block in rng.sample(sample, 8):
        user = first_user(block)
        print(f"    {user[:110]}")
    failed = corrupted_answers or corrupted_contexts or unchanged >= 0.1 * len(probe)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
