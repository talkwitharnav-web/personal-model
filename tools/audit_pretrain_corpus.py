"""Audit the tokenized pretraining corpus before committing 30 hours to it.

Checks the things that are cheap now and expensive later:

  1. Does the token stream decode back to sane text? A corpus can be structurally perfect and
     still be garbage. This prints real decoded windows to be read by eye.
  2. Is the unigram floor for THIS corpus and THIS tokenizer what the trainer thinks it is?
     The 7.4530 figure was measured on the old tokenizer and the old Wikipedia-only corpus.
     Reusing it against a different corpus would make the "model is not using context" alarm
     either silent or permanently on.
  3. Are all ids in range, is <|endoftext|> present at a sane rate, and did the source mix
     survive tokenization?
  4. Is there gross duplication - the same window repeated, which wastes budget and teaches
     memorisation.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from tools.memtrack import report

# The Windows console is cp1252 and cannot print most of what a Wikipedia corpus contains
# (this died on 'ł'). Decoded samples are the whole point of this audit, so force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path("data/pretrain")
TOKENIZER = Path("tokenizer/tokenizer_v2.json")
WINDOWS = 6
WINDOW_TOKENS = 90


def main():
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    vocab = tokenizer.get_vocab_size()
    endoftext = tokenizer.token_to_id("<|endoftext|>")

    train = np.memmap(DATA_DIR / "pretrain_train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(DATA_DIR / "pretrain_validation.bin", dtype=np.uint16, mode="r")
    print(f"train {len(train):,} tokens | validation {len(val):,} tokens | vocab {vocab:,}")

    failures = 0

    # ---- 1. ids in range -------------------------------------------------
    sample = np.asarray(train[:80_000_000])
    high = int(sample.max())
    print(f"\nmax token id in first 80M: {high:,}")
    if high >= vocab:
        print(f"  FAIL  id {high} >= vocab {vocab}")
        failures += 1
    else:
        print("  PASS  all ids inside the vocabulary")

    # ---- 2. document separator rate --------------------------------------
    eot = int((sample == endoftext).sum())
    per_doc = len(sample) / max(eot, 1)
    print(f"\n<|endoftext|> markers: {eot:,} -> mean {per_doc:,.0f} tokens/document")
    if eot == 0:
        print("  FAIL  no document separator in the stream")
        failures += 1
    else:
        print("  PASS  documents are separated")

    # ---- 3. unigram floor for THIS corpus --------------------------------
    counts = np.bincount(sample, minlength=vocab).astype(np.float64)
    probabilities = (counts + 1.0) / (counts.sum() + vocab)
    val_sample = np.asarray(val[:2_000_000])
    floor = float(-np.log(probabilities[val_sample]).mean())
    print(f"\nunigram floor for this corpus+tokenizer: {floor:.4f} nats")
    print(f"  uniform baseline ln({vocab}) = {np.log(vocab):.4f}")
    print(f"  (the old figure, measured on the OLD tokenizer and Wikipedia-only corpus: 7.4530)")

    # ---- 4. duplication --------------------------------------------------
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(train) - 64, size=40_000)
    fingerprints = Counter(train[s:s + 24].tobytes() for s in starts)
    repeats = sum(c - 1 for c in fingerprints.values() if c > 1)
    print(f"\nduplicate 24-token windows in 40,000 samples: {repeats} "
          f"({repeats/40_000*100:.3f}%)")
    if repeats / 40_000 > 0.05:
        print("  WARN  unusually high duplication")
    else:
        print("  PASS  no gross duplication")

    # ---- 5. eyeball it ---------------------------------------------------
    print("\n" + "=" * 76)
    print("DECODED WINDOWS - read these, structure proves nothing about quality")
    print("=" * 76)
    positions = rng.integers(0, len(train) - WINDOW_TOKENS, size=WINDOWS)
    for index, start in enumerate(positions, 1):
        ids = [int(t) for t in train[start:start + WINDOW_TOKENS]]
        text = tokenizer.decode(ids)
        print(f"\n--- window {index} @ token {start:,} ---")
        print(text.strip()[:520])

    print("\n" + "=" * 76)
    print(f"unigram floor to use in training/pretrain_v2.py: {floor:.4f}")
    report("(end of audit)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
