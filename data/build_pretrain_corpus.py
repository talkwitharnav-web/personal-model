"""Step 4b: interleave the cleaned corpora into ONE mixed pretraining corpus.

v1 pretrained in three sequential phases (TinyStories, then WikiText, then Wikipedia) and each
phase eroded the last - the same catastrophic forgetting that forced the fine-tune datasets to be
blended rather than trained one after another (README, onboarding section 5). So v2 uses a single
mixed corpus: documents from every source interleaved at random, which means any window the
trainer slices spans several sources instead of sitting inside one.

Proportions (by bytes):
  cleaned Wikipedia  80%  - the knowledge source, and what RAG actually retrieves at serving time
  TinyStories        15%  - simple, clean narrative English; cheap grammar for a small model
  WikiText            5%  - encyclopaedic prose, different register from the dump

Budgeting is done in BYTES with deliberate overshoot; the tokenize step enforces the exact token
target, exactly as prepare_wikipedia.py/tokenize_wikipedia.py already split that responsibility.

Memory: every source is a generator with its own small buffer and output is written incrementally.
Nothing here ever holds a whole corpus - this machine has 15.5 GB and Python dies silently when it
runs short (onboarding section 2). Peak RSS is reported at the end rather than assumed.
"""
import argparse
import random
import time
from pathlib import Path

from data.clean_wikipedia import iter_articles
from tools.memtrack import report

OUT_DIR = Path("data/pretrain")

TRAIN_SOURCES = {
    "wikipedia": (Path("data/pretrain/wikipedia_clean.txt"), 0.80),
    "tinystories": (Path("data/tinystories/train.txt"), 0.15),
    "wikitext": (Path("data/wikitext/train.txt"), 0.05),
}
VALIDATION_SOURCES = {
    "wikipedia": (Path("data/pretrain/wikipedia_clean_validation.txt"), 0.80),
    "tinystories": (Path("data/tinystories/validation.txt"), 0.15),
    "wikitext": (Path("data/wikitext/validation.txt"), 0.05),
}

MIN_DOC_CHARS = 120


def mix(sources, destination, target_bytes, seed=1234):
    rng = random.Random(seed)
    streams = {}
    for name, (path, weight) in sources.items():
        if not path.exists():
            print(f"  WARNING: missing {path}, skipping {name}")
            continue
        streams[name] = {"iterator": iter_articles(path), "weight": weight,
                         "docs": 0, "bytes": 0, "done": False}
    if not streams:
        raise SystemExit("no sources available")

    written = 0
    started = time.time()
    destination.parent.mkdir(parents=True, exist_ok=True)

    with open(destination, "w", encoding="utf-8") as out:
        while written < target_bytes:
            live = [n for n, s in streams.items() if not s["done"]]
            if not live:
                print("  all sources exhausted before the target was reached")
                break

            # Choose by BYTE deficit, not per document. Picking sources with fixed per-document
            # probability gives the wrong byte mix whenever documents differ in length: Wikipedia
            # articles average 2.88 KB and TinyStories 0.84 KB, so an 80% document share came out
            # as 94% of bytes. Weighting by how far each source is BELOW its byte target converges
            # on the intended proportions regardless of document size.
            total_live_weight = sum(streams[n]["weight"] for n in live)
            deficits = []
            for name in live:
                share = streams[name]["weight"] / total_live_weight
                deficit = share * max(written, 1) - streams[name]["bytes"]
                deficits.append(max(deficit, 0.0))
            if sum(deficits) <= 0:
                deficits = [streams[n]["weight"] for n in live]
            picked = rng.choices(live, weights=deficits)[0]

            stream = streams[picked]
            try:
                document = next(stream["iterator"])
            except StopIteration:
                stream["done"] = True
                print(f"  {picked} exhausted after {stream['docs']:,} documents")
                continue
            if len(document) < MIN_DOC_CHARS:
                continue

            out.write(document + "\n<|endoftext|>\n")
            stream["docs"] += 1
            stream["bytes"] += len(document)
            written += len(document)

            if written // (512 * 1024 * 1024) != (written - len(document)) // (512 * 1024 * 1024):
                rate = written / (time.time() - started) / 1024 ** 2
                print(f"  {written/1024**3:.2f} GB written | {rate:.0f} MB/s")

    elapsed = time.time() - started
    print(f"\n{destination.name}  ({elapsed/60:.1f} min)")
    total_docs = sum(s["docs"] for s in streams.values())
    for name, stream in streams.items():
        share = stream["bytes"] / max(1, written) * 100
        print(f"  {name:<12} {stream['docs']:>9,} docs | {stream['bytes']/1024**3:>6.2f} GB | "
              f"{share:>5.1f}%  (target {sources[name][1]*100:.0f}%)")
    print(f"  {'TOTAL':<12} {total_docs:>9,} docs | {written/1024**3:>6.2f} GB")
    report("(after mixing)")
    return written


def main():
    parser = argparse.ArgumentParser()
    # 1.6B tokens at ~4.1 bytes/token = 6.6 GB; 7.4 GB is overshoot so the tokenizer can hit its
    # exact token target without running out of text.
    parser.add_argument("--target-gb", type=float, default=7.4)
    # Capped by the smallest source: the cleaned Wikipedia validation file is only ~10 MB, so at
    # an 80% share the total cannot exceed ~12.5 MB without Wikipedia running dry and the mix
    # silently inverting (the first run came out 43% Wikipedia / 54% TinyStories for exactly
    # this reason). ~12 MB is about 3M tokens, matching the old validation budget.
    parser.add_argument("--validation-mb", type=float, default=12)
    args = parser.parse_args()

    print("=== building the mixed v2 pretraining corpus ===")
    mix(TRAIN_SOURCES, OUT_DIR / "pretrain_train.txt", int(args.target_gb * 1024 ** 3))
    print()
    mix(VALIDATION_SOURCES, OUT_DIR / "pretrain_validation.txt",
        int(args.validation_mb * 1024 ** 2), seed=99)


if __name__ == "__main__":
    main()
