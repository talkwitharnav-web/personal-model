"""How much of the RAG index is junk the Wikipedia cleaner would have removed?

`rag/build_index.py` reads `data/wikipedia/train.txt` - the RAW dump. The index was built on
2026-08-15; `data/clean_wikipedia.py` was written on 2026-08-17 and drops 305,368 articles (10.5%)
as disambiguation pages, list stubs and template wreckage. None of that cleaning reached the index.

It shows up directly in answers. Asked "What is the capital of France?", the retriever returns:

    0.726  'Paris is the capital of France, which may consist of :'      <- disambiguation stub
    0.667  'It may also refer to the following places in France:'        <- boilerplate
    0.667  'It may also refer to the following places in France:'        <- the SAME passage twice

The fact is technically present, buried in a disambiguation header and padded with two identical
junk passages, and the model answered "Founded in the 19th century".

This counts how common that is, so the cost of rebuilding the index can be judged on evidence.
"""
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PASSAGES = Path("rag/wikipedia_passages.txt")

# Phrases that mark a passage as navigational rather than informational.
JUNK_PATTERNS = [
    (r"may refer to", "'may refer to' disambiguation"),
    (r"may also refer to", "'may also refer to' boilerplate"),
    (r"which may consist of", "'may consist of' stub"),
    (r"^\s*(see also|references|external links|further reading|notes)\s*$", "section heading only"),
    (r"is a (surname|given name|family name)", "name-list page"),
    (r"^\s*(list of|index of)", "list page"),
    (r"\bdisambiguation\b", "explicit disambiguation"),
]


def main():
    if not PASSAGES.exists():
        raise SystemExit(f"missing {PASSAGES}")

    limit = 400_000
    counts = Counter()
    seen = Counter()
    total = 0
    compiled = [(re.compile(p, re.I | re.M), label) for p, label in JUNK_PATTERNS]

    with open(PASSAGES, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            seen[line] += 1
            for pattern, label in compiled:
                if pattern.search(line):
                    counts[label] += 1
                    break
            if total >= limit:
                break

    junk = sum(counts.values())
    duplicated = sum(count - 1 for count in seen.values() if count > 1)

    print(f"sampled {total:,} passages from {PASSAGES}\n")
    print(f"  junk passages:      {junk:,}  ({100*junk/total:.1f}%)")
    for label, count in counts.most_common():
        print(f"      {label:<34} {count:>7,}")
    print(f"\n  duplicate passages: {duplicated:,}  ({100*duplicated/total:.1f}% of the sample)")
    print(f"  distinct passages:  {len(seen):,}")

    worst = [(text, count) for text, count in seen.most_common(5)]
    print("\n  most repeated passages:")
    for text, count in worst:
        print(f"      x{count:<5} {text[:88]}")

    print("\nA duplicate wastes one of the three retrieval slots, so a query that pulls the same")
    print("boilerplate twice has only one real passage left to answer from.")
    print("\nFIX: point WIKIPEDIA_PATH at data/pretrain/wikipedia_clean.txt and rebuild, and")
    print("de-duplicate passages before indexing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
