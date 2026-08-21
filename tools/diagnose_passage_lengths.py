"""Did cleaning delete the SHORT definitional passages? Compare length distributions.

The cleaned index ranks worse at hit@1 (40% vs 68%, McNemar p=0.0391) and the top-1 passages show
a clear pattern - the old index returns definitions, the new one returns passing mentions:

    What is DNA?   old: "DNA (deoxyribonucleic acid) is a molecule encoding the genetic
                        instructions for life."                              (84 characters)
                   new: "DNA is used to describe the skills, properties or qualities..."

The hypothesis is that `MIN_ARTICLE_CHARS = 200` in `data/clean_wikipedia.py` deleted exactly the
short definitional stubs that answer "what is X" best - a length filter cannot tell a junk stub
from a perfect one-line definition.

Two earlier attempts to confirm this failed for sloppy reasons: substring probes that were too
strict, and scans that silently covered only 5.6 GB of 16 GB files. This measures the thing the
hypothesis actually predicts instead of hunting individual strings.

PREDICTION: if the hypothesis holds, the cleaned passage file contains a markedly smaller SHARE of
short passages than the raw one. If the two length distributions are similar, the hypothesis is
refuted and the remaining candidate is quantisation - `IndexIVFPQ` retraining its centroids on a
different corpus.

Sampling is by random byte offset across the whole file (seek, discard the partial line, take the
next one), so it is unbiased with respect to position. Both files are ~16 GB; nothing is loaded
whole.
"""
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FILES = {
    "OLD (raw dump)": Path("rag/wikipedia_passages.txt"),
    "NEW (cleaned)": Path("rag/wikipedia_clean_passages.txt"),
}
SAMPLES = 60_000
BUCKETS = [(0, 100), (100, 150), (150, 200), (200, 300), (300, 500), (500, 10 ** 9)]


def sample_lengths(path, samples, seed=7):
    size = path.stat().st_size
    rng = random.Random(seed)
    lengths = []
    with open(path, "rb") as handle:
        for _ in range(samples):
            handle.seek(rng.randrange(0, size))
            handle.readline()                 # discard the partial line landed in
            line = handle.readline()
            if not line:
                continue
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            if text:
                lengths.append(len(text))
    return lengths


def main():
    results = {}
    for label, path in FILES.items():
        if not path.exists():
            raise SystemExit(f"missing {path}")
        lengths = sample_lengths(path, SAMPLES)
        lengths.sort()
        results[label] = lengths
        n = len(lengths)
        mean = sum(lengths) / n
        median = lengths[n // 2]
        print(f"\n{label}  ({path.stat().st_size / 1024**3:.1f} GB, {n:,} sampled passages)")
        print(f"  mean {mean:.0f} chars | median {median} chars")
        print(f"  {'length':<14}{'count':>9}{'share':>9}")
        for low, high in BUCKETS:
            count = sum(1 for length in lengths if low <= length < high)
            label_range = f"{low}-{high}" if high < 10 ** 9 else f"{low}+"
            print(f"  {label_range:<14}{count:>9,}{100*count/n:>8.1f}%")

    print(f"\n{'=' * 88}\nVERDICT\n{'=' * 88}")
    labels = list(FILES)
    old, new = results[labels[0]], results[labels[1]]
    old_short = 100 * sum(1 for x in old if x < 200) / len(old)
    new_short = 100 * sum(1 for x in new if x < 200) / len(new)
    print(f"  passages under 200 characters:  old {old_short:.1f}%   new {new_short:.1f}%")
    gap = old_short - new_short
    if gap > 5:
        print(f"\n  -> SUPPORTED. Cleaning removed {gap:.1f} points of short passages. The")
        print("     MIN_ARTICLE_CHARS=200 filter is deleting short definitional stubs along with")
        print("     the junk, which is the most likely reason hit@1 fell.")
        print("     Cheap test of the fix: re-clean with MIN_ARTICLE_CHARS≈50 and rebuild.")
    elif abs(gap) <= 5:
        print(f"\n  -> REFUTED. The two distributions are within {abs(gap):.1f} points, so short")
        print("     passages were NOT preferentially removed. The length-filter story is wrong;")
        print("     the remaining candidate is IndexIVFPQ retraining its centroids on a different")
        print("     corpus. Do not repeat the length explanation.")
    else:
        print(f"\n  -> The CLEANED file has MORE short passages ({-gap:.1f} points). The length")
        print("     hypothesis is refuted outright.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
