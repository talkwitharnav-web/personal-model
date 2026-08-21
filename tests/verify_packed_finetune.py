"""Verify the packed fine-tune data before training on it.

Onboarding section 6: "Verify before training, always: check for rows where every label is -100
(zero valid targets -> NaN loss), and confirm unmasked labels equal the ids at the same positions
(catches off-by-one masking)."

Packing adds two failure modes of its own:
  - a row could land entirely inside one example's masked PROMPT, giving zero valid targets and a
    NaN loss for that step
  - the ids and labels streams could drift apart by a token, which would train every position
    against its neighbour's target and still look completely normal in aggregate

Both are silent. Neither shows up in a loss curve until the run is wasted.
"""
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

DIR = Path("data/finetune_packed")
TOKENIZER = Path("tokenizer/tokenizer_v2.json")
CONTEXT = 768

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}  {detail}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    ids = np.load(DIR / f"{split}_ids.npy", mmap_mode="r")
    labels = np.load(DIR / f"{split}_labels.npy", mmap_mode="r")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    endoftext = tokenizer.token_to_id("<|endoftext|>")

    print(f"=== {split}: {ids.shape[0]:,} rows x {ids.shape[1]} ===")
    check("ids and labels have the same shape", ids.shape == labels.shape,
          f"{ids.shape} vs {labels.shape}")
    check("row width is the context length", ids.shape[1] == CONTEXT, str(ids.shape[1]))

    rng = np.random.default_rng(0)
    sample = rng.choice(ids.shape[0], size=min(3000, ids.shape[0]), replace=False)

    # THE off-by-one check: wherever a label is not -100 it must equal the id at that position.
    mismatches = 0
    all_masked = 0
    valid_fraction = []
    for row in sample:
        row_ids = np.asarray(ids[row])
        row_labels = np.asarray(labels[row])
        keep = row_labels != -100
        if not keep.any():
            all_masked += 1
            continue
        if not np.array_equal(row_ids[keep], row_labels[keep]):
            mismatches += 1
        valid_fraction.append(float(keep.mean()))

    check("unmasked labels equal ids at the same positions", mismatches == 0,
          f"{mismatches} rows differ (off-by-one masking would show here)")
    check("no row is entirely masked (-100 everywhere -> NaN loss)", all_masked == 0,
          f"{all_masked} rows")

    if valid_fraction:
        mean_valid = sum(valid_fraction) / len(valid_fraction)
        print(f"  positions carrying gradient: {mean_valid*100:.1f}% mean per row")
        print(f"    (v1: ~2.8% of positions, because 71.3% of every row was padding)")
        check("a useful share of positions carry gradient", mean_valid > 0.05,
              f"{mean_valid*100:.1f}%")

    # Ids should be inside the vocabulary; labels are ids or -100.
    flat_ids = np.asarray(ids[sample[:400]])
    flat_labels = np.asarray(labels[sample[:400]])
    check("all ids inside the vocabulary", int(flat_ids.max()) < tokenizer.get_vocab_size()
          and int(flat_ids.min()) >= 0, f"[{flat_ids.min()}, {flat_ids.max()}]")
    bad_labels = int(((flat_labels != -100) & (flat_labels < 0)).sum())
    check("labels are either a real id or exactly -100", bad_labels == 0, f"{bad_labels} bad")

    # Packing should put several examples in a row - check the separator actually appears.
    separators = int((np.asarray(ids[sample[:400]]) == endoftext).sum()) / 400
    print(f"  <|endoftext|> separators per row: {separators:.2f}")
    check("rows contain example boundaries", separators > 0.5, f"{separators:.2f} per row")

    print("\n--- one decoded row, read it ---")
    row = int(sample[0])
    text = tokenizer.decode([int(t) for t in ids[row]])
    print(text[:700].replace("\n", "\\n")[:700])

    print("\n--- the same row's TRAINED region only (labels != -100) ---")
    row_ids = np.asarray(ids[row])
    row_labels = np.asarray(labels[row])
    trained = tokenizer.decode([int(t) for t in row_ids[row_labels != -100]])
    print(trained[:500].replace("\n", "\\n")[:500])

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
