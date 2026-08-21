"""Measure what junk is actually in data/wikipedia/train.txt before writing cleaning rules.

Guessing at cleaning rules and then deleting 18 GB on the strength of the guess is how you find
out afterwards that you threw away something you needed. This reads a sample, counts each
category, and prints real examples of every one so the rules can be written against evidence.

Streams in binary and stays in bytes (onboarding section 7 - a byte-vs-char mixup silently
truncated 17,162 articles once).
"""
import re
import sys
from collections import Counter

CORPUS = "data/wikipedia/train.txt"
SAMPLE_BYTES = 400 * 1024 * 1024
SHOW = 4

PATTERNS = {
    "redirect": re.compile(r"^\s*#?\s*REDIRECT\b", re.I),
    "disambiguation": re.compile(r"\bmay refer to\s*:|\bmay also refer to\s*:|"
                                 r"\bis a disambiguation page", re.I),
    "list_article": re.compile(r"^\s*List of\b|^\s*Index of\b|^\s*Outline of\b", re.I),
    "empty_parens": re.compile(r"\(\s*[,;]?\s*\)"),
    "dangling_unit": re.compile(r"\b(?:of|at|is|was|about|approximately)\s+[,.]"),
    "empty_convert": re.compile(r"\s+\(\s*\)|\s+\[\s*\]"),
    "reference_line": re.compile(r"\bISBN\b|\bdoi:|\bOCLC\b|\bJSTOR\b|\barXiv\b|"
                                 r"Archived from the original|Retrieved \d"),
    "section_stub": re.compile(r"^\s*(?:See also|References|External links|Further reading|"
                               r"Notes|Sources|Bibliography|Citations)\s*$", re.I),
    "url": re.compile(r"https?://|www\."),
}


def main():
    counts = Counter()
    examples = {name: [] for name in PATTERNS}
    line_counts = Counter()
    line_examples = {name: [] for name in PATTERNS}

    with open(CORPUS, "rb") as handle:
        raw = handle.read(SAMPLE_BYTES)
    cut = raw.rfind(b"<|endoftext|>")
    raw = raw[:cut] if cut != -1 else raw
    text = raw.decode("utf-8", errors="ignore")

    articles = [a.strip() for a in text.split("<|endoftext|>") if a.strip()]
    print(f"sample: {len(raw):,} bytes -> {len(articles):,} articles\n")

    total_lines = 0
    lengths = []
    for article in articles:
        lengths.append(len(article))
        head = article[:400]
        for name, pattern in PATTERNS.items():
            if pattern.search(head if name in ("redirect", "disambiguation", "list_article")
                              else article):
                counts[name] += 1
                if len(examples[name]) < SHOW:
                    match = pattern.search(article)
                    start = max(0, match.start() - 90)
                    examples[name].append(article[start:match.end() + 90].replace("\n", " | "))
        for line in article.split("\n"):
            total_lines += 1
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    line_counts[name] += 1
                    if len(line_examples[name]) < SHOW:
                        line_examples[name].append(line.strip()[:200])

    print("ARTICLE-LEVEL (how many articles contain this at all)")
    print("-" * 74)
    for name in PATTERNS:
        pct = counts[name] / len(articles) * 100
        print(f"  {name:<16} {counts[name]:>7,}  {pct:>6.2f}% of articles")

    print(f"\nLINE-LEVEL ({total_lines:,} lines total)")
    print("-" * 74)
    for name in PATTERNS:
        pct = line_counts[name] / total_lines * 100
        print(f"  {name:<16} {line_counts[name]:>7,}  {pct:>6.2f}% of lines")

    lengths.sort()
    print(f"\nARTICLE LENGTH (bytes): p10 {lengths[len(lengths)//10]:,} | "
          f"median {lengths[len(lengths)//2]:,} | p90 {lengths[len(lengths)*9//10]:,}")
    print(f"  under 200 chars: {sum(1 for l in lengths if l < 200):,} "
          f"({sum(1 for l in lengths if l < 200)/len(lengths)*100:.2f}%)")
    print(f"  under 500 chars: {sum(1 for l in lengths if l < 500):,} "
          f"({sum(1 for l in lengths if l < 500)/len(lengths)*100:.2f}%)")

    print("\n\nREAL EXAMPLES")
    print("=" * 74)
    for name in PATTERNS:
        print(f"\n--- {name} (line level) ---")
        if not line_examples[name]:
            print("    (none found)")
        for sample in line_examples[name]:
            print(f"    {sample!r}")


if __name__ == "__main__":
    sys.exit(main())
