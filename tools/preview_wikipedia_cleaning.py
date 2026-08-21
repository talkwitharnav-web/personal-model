"""Eyeball the cleaning rules on real articles BEFORE running them over 18.9 GB.

The danger with a repair-and-drop cleaner is not that it fails loudly, it is that it quietly
deletes good prose. An earlier version of these rules rejected any line containing wiki markup
and threw away real biography sentences for a stray tag near the end - that was caught here, by
printing what was dropped instead of trusting the rule.

So this reports what is dropped, what is repaired, and shows before/after text for both.
"""
import re
from collections import Counter

from data.clean_wikipedia import (
    BARE_HEADING, BIBLIOGRAPHY, EMPTY_BRACKETS, STRIPPED_VALUE, URL,
    clean_article, clean_line,
)
from tools.memtrack import report

SAMPLE_BYTES = 120 * 1024 * 1024
SHOW = 5

HAD_MARKUP = re.compile(r"\{\{|\[\[|<ref\b|</ref>")


def main():
    with open("data/wikipedia/train.txt", "rb") as handle:
        raw = handle.read(SAMPLE_BYTES)
    cut = raw.rfind(b"<|endoftext|>")
    text = raw[:cut].decode("utf-8", errors="ignore")
    del raw                                    # 120 MB of bytes, no longer needed
    articles = [a.strip() for a in text.split("<|endoftext|>") if a.strip()]
    del text                                   # and 120 MB of str
    print(f"sample: {len(articles):,} articles")
    report("(after loading sample)")
    print()

    dropped = Counter()
    repaired = Counter()
    examples = {"dropped_line": [], "markup_kept": [], "url_kept": [], "brackets": [],
                "stripped_sentence": []}
    total_lines = 0

    for article in articles:
        for line in article.split("\n"):
            if not line.strip():
                continue
            total_lines += 1
            had_markup = bool(HAD_MARKUP.search(line))
            had_url = bool(URL.search(line))
            had_brackets = bool(EMPTY_BRACKETS.search(line))
            had_stripped = bool(STRIPPED_VALUE.search(line))

            cleaned = clean_line(line)

            if not cleaned:
                if BARE_HEADING.match(line):
                    dropped["bare_heading"] += 1
                elif BIBLIOGRAPHY.search(line):
                    dropped["bibliography"] += 1
                else:
                    dropped["residue_after_stripping"] += 1
                    if len(examples["dropped_line"]) < SHOW:
                        examples["dropped_line"].append(line.strip()[:150])
                continue

            if had_markup:
                repaired["markup_stripped"] += 1
                if len(examples["markup_kept"]) < SHOW:
                    examples["markup_kept"].append((line.strip()[:130], cleaned[:130]))
            if had_url:
                repaired["url_stripped"] += 1
                if len(examples["url_kept"]) < SHOW:
                    examples["url_kept"].append((line.strip()[:130], cleaned[:130]))
            if had_brackets:
                repaired["empty_brackets"] += 1
                if len(examples["brackets"]) < SHOW:
                    examples["brackets"].append((line.strip()[:130], cleaned[:130]))
            if had_stripped:
                repaired["broken_sentence_removed"] += 1
                if len(examples["stripped_sentence"]) < SHOW:
                    examples["stripped_sentence"].append((line.strip()[:130], cleaned[:130]))

    print(f"LINES DROPPED ENTIRELY, out of {total_lines:,}")
    print("-" * 74)
    for name, count in dropped.most_common():
        print(f"  {name:<26} {count:>8,}  {count/total_lines*100:>6.3f}%")
    print(f"  {'TOTAL':<26} {sum(dropped.values()):>8,}  "
          f"{sum(dropped.values())/total_lines*100:>6.3f}%")

    print("\nLINES KEPT BUT REPAIRED")
    print("-" * 74)
    for name, count in repaired.most_common():
        print(f"  {name:<26} {count:>8,}  {count/total_lines*100:>6.3f}%")

    print("\n\nPROSE THAT USED TO BE DELETED IS NOW KEPT (before -> after)")
    print("=" * 74)
    for label in ("markup_kept", "url_kept"):
        print(f"\n--- {label} ---")
        for before, after in examples[label]:
            print(f"  BEFORE: {before!r}")
            print(f"  AFTER : {after!r}\n")

    print("\nREPAIRS")
    print("=" * 74)
    for label in ("brackets", "stripped_sentence"):
        print(f"\n--- {label} ---")
        for before, after in examples[label]:
            print(f"  BEFORE: {before!r}")
            print(f"  AFTER : {after!r}\n")

    print("\nLINES STILL DROPPED AS RESIDUE - check none of this is good prose")
    print("=" * 74)
    for sample in examples["dropped_line"] or ["(none)"]:
        print(f"    {sample!r}")

    # One pass, not three: clean_article is not free and was being called repeatedly per article.
    verdicts = [clean_article(a) for a in articles]
    kept = sum(1 for cleaned, _ in verdicts if cleaned is not None)
    reasons = Counter(reason for cleaned, reason in verdicts if cleaned is None)
    print(f"\n\nARTICLE LEVEL: {kept:,}/{len(articles):,} kept ({kept/len(articles)*100:.1f}%)")
    for reason, count in reasons.most_common():
        print(f"    dropped {reason}: {count:,}")
    report("(end of preview)")


if __name__ == "__main__":
    main()
