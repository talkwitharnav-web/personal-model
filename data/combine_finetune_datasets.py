import argparse
from pathlib import Path

import numpy as np

from tools.memtrack import report

# Memory-safe rewrite. The original loaded every dataset fully (no mmap_mode), built two lists,
# concatenated them (doubling), then fancy-indexed to shuffle (doubling again) - roughly four
# copies of the payload alive at once:
#
#   571,319 rows x 768 x 4 bytes = 1.755 GB per array, ids + labels = 3.5 GB
#   peak during concatenate + shuffle ~= 7-9 GB of this machine's 15.5 GB
#
# It survived on the v1 mix. The v2 mix adds GSM8K, SODA, No Robots and 5% replay - about 750k
# rows, roughly 11-12 GB peak, which is exactly the profile of the silent Python deaths in
# onboarding section 2: no traceback, the process simply vanishes.
#
# This version never holds more than one chunk. It memory-maps the sources, preallocates the
# output with open_memmap, and copies rows straight to their shuffled destinations.
#
# Output is IDENTICAL to the original: same seed, same caps, same permutation. The original
# gathered (dest[i] = src[order[i]]); doing that directly means random reads, so this scatters
# with the inverse permutation (dest[inverse[s]] = src[s]) - the same mapping, but the sources
# are read sequentially.

dataset_names = [
    "commonsenseqa", "sciq", "piqa", "boolq", "hellaswag", "socialiqa",
    "logiqa", "reclor", "squad", "race", "xsum", "oasst1", "personachat",
    "proofwriter", "definitions", "yesno",
    # See data/pack_finetune_datasets.py - GSM8K joins the v2 mix per section 10.
    "gsm8k_clean", "soda", "no_robots", "replay",
    # Messy-input copies (typos, shorthand, no capitals) - see data/augment_typos.py.
    "typo_aug",
    # Bare greetings and command-shaped instructions - see data/prepare_greetings.py and
    # data/prepare_instruction_shapes.py.
    "greetings", "instruction_shapes",
]

# XSum was 157,923 examples - 27.6% of the whole mix - so more than a quarter of
# training was "summarise a news article", crowding out the question answering.
MAX_PER_DATASET = {"xsum": 50_000, "typo_aug": 60_000}

ROWS_PER_CHUNK = 4096
# NO periodic flush. It seems like it should help - dirty pages of a memory-mapped file count
# toward the working set - but MEASURED IT MAKES THINGS WORSE: flushing the whole 1.75 GB mapping
# every few chunks forces the OS to walk it and pull pages back in. Train split peaked at
# 2,572 MB without flushing and 3,476 MB with it. Left out on the evidence.


def plan(split_name):
    """Work out every source's length and dtype WITHOUT reading the data.

    mmap_mode="r" reads only the .npy header, so this costs nothing across all 16 datasets.
    """
    entries = []
    total = 0
    ids_dtype = labels_dtype = None
    width = None
    for name in dataset_names:
        ids = np.load(f"data/{name}/{split_name}_ids.npy", mmap_mode="r")
        labels = np.load(f"data/{name}/{split_name}_labels.npy", mmap_mode="r")
        if ids.shape != labels.shape:
            raise SystemExit(f"{name}: ids {ids.shape} != labels {labels.shape}")

        keep = None
        cap = MAX_PER_DATASET.get(name)
        if cap and len(ids) > cap and split_name == "train":
            # Same rng and same call as the original, so the same rows are kept.
            keep = np.random.default_rng(0).choice(len(ids), cap, replace=False)
            length = cap
        else:
            length = len(ids)

        if ids_dtype is None:
            ids_dtype, labels_dtype, width = ids.dtype, labels.dtype, ids.shape[1]
        elif ids.dtype != ids_dtype or labels.dtype != labels_dtype or ids.shape[1] != width:
            raise SystemExit(f"{name}: dtype/width mismatch "
                             f"({ids.dtype},{labels.dtype},{ids.shape[1]}) vs "
                             f"({ids_dtype},{labels_dtype},{width})")

        entries.append({"name": name, "keep": keep, "length": length, "offset": total})
        total += length
        print(f"{name}: {length:,} x {width}"
              + (f"  (capped from {len(ids):,})" if keep is not None else ""))
        del ids, labels

    return entries, total, width, ids_dtype, labels_dtype


def combine_split(split_name, output_dir):
    entries, total, width, ids_dtype, labels_dtype = plan(split_name)
    print(f"Combined {split_name}: ({total:,}, {width})")

    # Same permutation as the original: dest[i] = src[order[i]].
    rng = np.random.default_rng(seed=42)
    order = rng.permutation(total)
    # Inverted so the sources can be read sequentially and scattered into place instead.
    inverse = np.empty_like(order)
    inverse[order] = np.arange(total)
    del order

    output_dir.mkdir(parents=True, exist_ok=True)
    out_ids = np.lib.format.open_memmap(
        output_dir / f"{split_name}_ids.npy", mode="w+", dtype=ids_dtype, shape=(total, width))
    out_labels = np.lib.format.open_memmap(
        output_dir / f"{split_name}_labels.npy", mode="w+", dtype=labels_dtype,
        shape=(total, width))

    chunks_written = 0
    for entry in entries:
        name, keep, length, offset = (entry["name"], entry["keep"],
                                      entry["length"], entry["offset"])
        src_ids = np.load(f"data/{name}/{split_name}_ids.npy", mmap_mode="r")
        src_labels = np.load(f"data/{name}/{split_name}_labels.npy", mmap_mode="r")

        for start in range(0, length, ROWS_PER_CHUNK):
            stop = min(start + ROWS_PER_CHUNK, length)
            rows = keep[start:stop] if keep is not None else slice(start, stop)
            destination = inverse[offset + start:offset + stop]
            # One chunk in memory at a time: 4096 x 768 x 4 bytes = 12.6 MB.
            out_ids[destination] = src_ids[rows]
            out_labels[destination] = src_labels[rows]
            chunks_written += 1

        del src_ids, src_labels
        print(f"  placed {name}: {length:,} rows")

    out_ids.flush()
    out_labels.flush()
    del out_ids, out_labels
    print(f"Saved combined {split_name} set: ({total:,}, {width})")
    report(f"(after {split_name})")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/combined",
                        help="output directory - point at a scratch dir to avoid overwriting v1")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"],
                        choices=["train", "validation"])
    args = parser.parse_args()
    output_dir = Path(args.out)

    for split_name in args.splits:
        combine_split(split_name, output_dir)


if __name__ == "__main__":
    main()
