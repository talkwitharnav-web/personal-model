"""Prove the memory-safe combine produces EXACTLY what the original produced.

A rewrite that halves memory but quietly reorders rows, or pairs an example's tokens with another
example's mask, would be far worse than the bug it fixes - and it would not announce itself. Loss
would just be mysteriously bad.

So this does not re-run the original (which is the whole point - it needs 7-9 GB). It checks the
original's defining property directly against the new output:

    dest[i] == concat(sources)[order[i]]     where order = default_rng(42).permutation(total)

reading single rows from the memory-mapped sources, which costs nothing. It also verifies that
ids and labels were permuted with the SAME order - if they ever came apart, every example would
be trained against another example's mask.
"""
import sys
from pathlib import Path

import numpy as np

from data.combine_finetune_datasets import plan

SAMPLES = 400


def check_split(split_name, output_dir, rng):
    entries, total, width, _, _ = plan(split_name)

    out_ids = np.load(output_dir / f"{split_name}_ids.npy", mmap_mode="r")
    out_labels = np.load(output_dir / f"{split_name}_labels.npy", mmap_mode="r")
    print(f"\n=== {split_name} ===")
    print(f"  output shape {out_ids.shape} (expected ({total:,}, {width}))")
    if out_ids.shape != (total, width) or out_labels.shape != (total, width):
        print("  FAIL  shape mismatch")
        return False

    order = np.random.default_rng(seed=42).permutation(total)

    sources = {}
    for entry in entries:
        sources[entry["name"]] = (
            np.load(f"data/{entry['name']}/{split_name}_ids.npy", mmap_mode="r"),
            np.load(f"data/{entry['name']}/{split_name}_labels.npy", mmap_mode="r"))

    def locate(global_index):
        for entry in entries:
            if entry["offset"] <= global_index < entry["offset"] + entry["length"]:
                local = global_index - entry["offset"]
                row = int(entry["keep"][local]) if entry["keep"] is not None else local
                return entry["name"], row
        raise IndexError(global_index)

    mismatches = 0
    label_mismatches = 0
    targets = rng.choice(total, size=min(SAMPLES, total), replace=False)
    for destination in targets:
        name, row = locate(int(order[destination]))
        src_ids, src_labels = sources[name]
        if not np.array_equal(out_ids[destination], src_ids[row]):
            mismatches += 1
        if not np.array_equal(out_labels[destination], src_labels[row]):
            label_mismatches += 1

    print(f"  checked {len(targets)} random destination rows against the original mapping")
    if mismatches:
        print(f"  FAIL  {mismatches} ids rows differ from dest[i] == src[order[i]]")
    else:
        print("  PASS  every sampled row matches the original's exact placement")
    if label_mismatches:
        print(f"  FAIL  {label_mismatches} labels rows differ")
    else:
        print("  PASS  labels landed with their own ids (masks not torn from examples)")

    # A mask that is entirely -100 gives a row with no valid target -> NaN loss.
    empty = 0
    for destination in targets[:120]:
        if int((out_labels[destination] != -100).sum()) == 0:
            empty += 1
    if empty:
        print(f"  WARN  {empty} sampled rows have every label masked (-100) -> NaN loss risk")
    else:
        print("  PASS  no fully-masked rows in the sample")

    return mismatches == 0 and label_mismatches == 0


def main():
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/combined")
    splits = sys.argv[2:] or ["validation"]
    rng = np.random.default_rng(0)
    ok = True
    for split_name in splits:
        if not (output_dir / f"{split_name}_ids.npy").exists():
            print(f"skipping {split_name}: not present in {output_dir}")
            continue
        ok = check_split(split_name, output_dir, rng) and ok
    print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
