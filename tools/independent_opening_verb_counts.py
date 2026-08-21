"""Count first words on User: lines without loading whole training files."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer


USER_LINE = re.compile(rb"(?m)^User:[ \t]*([A-Za-z][A-Za-z'-]*)")
USER_LINE_TEXT = re.compile(r"(?m)^User:[ \t]*([A-Za-z][A-Za-z'-]*)")
CHUNK = 8 * 1024 * 1024

DECLARED_MIX = [
    "commonsenseqa", "sciq", "piqa", "boolq", "hellaswag", "socialiqa",
    "logiqa", "reclor", "squad", "race", "xsum", "oasst1", "personachat",
    "proofwriter", "definitions", "yesno", "gsm8k_clean", "soda", "no_robots",
    "replay", "typo_aug", "greetings", "instruction_shapes",
]
CAPS = {"xsum": 50_000, "typo_aug": 60_000}


def has_user_lines(path: Path) -> bool:
    with path.open("rb") as handle:
        return bool(re.search(rb"(?m)^User:", handle.read(64 * 1024)))


def count_file(path: Path) -> Counter:
    counts: Counter = Counter()
    carry = b""
    with path.open("rb") as handle:
        while data := handle.read(CHUNK):
            data = carry + data
            split_at = data.rfind(b"\n")
            if split_at < 0:
                carry = data
                continue
            complete, carry = data[: split_at + 1], data[split_at + 1 :]
            counts.update(match.group(1).decode("ascii").lower()
                          for match in USER_LINE.finditer(complete))
    if carry:
        counts.update(match.group(1).decode("ascii").lower()
                      for match in USER_LINE.finditer(carry))
    return counts


def count_effective_mix(checkpoint_proxy: bool) -> tuple[Counter, dict]:
    """Reapply current packing caps and the 768-token length filter to source text."""
    from data.pack_finetune_datasets import iter_examples

    tokenizer = Tokenizer.from_file("tokenizer/tokenizer_v2.json")
    checkpoint_time = Path("checkpoints/finetune_v2_best.pt").stat().st_mtime
    total: Counter = Counter()
    by_dataset = {}
    for name in DECLARED_MIX:
        path = Path("data") / name / "train.txt"
        if not path.exists():
            continue
        if checkpoint_proxy and path.stat().st_mtime > checkpoint_time:
            continue
        counts: Counter = Counter()
        kept = 0
        for text in iter_examples(path, CAPS.get(name), random.Random(42)):
            if len(tokenizer.encode(text).ids) + 1 > 768:
                continue
            kept += 1
            counts.update(match.group(1).lower()
                          for match in USER_LINE_TEXT.finditer(text))
        total.update(counts)
        by_dataset[name] = {"documents": kept, "user_turns": sum(counts.values())}
    return total, by_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--words", nargs="*", default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--effective", action="store_true",
                        help="also emulate current packing caps and length filtering")
    args = parser.parse_args()

    by_dataset = {}
    total: Counter = Counter()
    contributions = defaultdict(Counter)
    for path in sorted(Path("data").glob("*/train.txt")):
        if not has_user_lines(path):
            continue
        counts = count_file(path)
        by_dataset[path.parent.name] = sum(counts.values())
        total.update(counts)
        for word, count in counts.items():
            contributions[word][path.parent.name] = count

    words = [word.lower() for word in args.words]
    result = {
        "datasets": by_dataset,
        "total_user_turns": sum(total.values()),
        "top_50": total.most_common(50),
        "selected": {
            word: {
                "count": total[word],
                "by_dataset": contributions[word].most_common(),
            }
            for word in words
        },
    }
    if args.effective:
        for label, proxy in (("effective_current_mix", False),
                             ("effective_checkpoint_time_proxy", True)):
            effective, effective_datasets = count_effective_mix(proxy)
            result[label] = {
                "datasets": effective_datasets,
                "total_user_turns": sum(effective.values()),
                "selected": {word: effective[word] for word in words},
            }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{len(by_dataset)} datasets, {result['total_user_turns']:,} User turns")
        print("\nTop openings:")
        for word, count in result["top_50"]:
            print(f"  {word:<18} {count:>9,}")
        if words:
            print("\nSelected:")
            for word in words:
                details = result["selected"][word]
                sources = ", ".join(f"{name}={count:,}"
                                    for name, count in details["by_dataset"][:8])
                print(f"  {word:<18} {details['count']:>9,}  {sources}")
        if args.effective:
            for label in ("effective_current_mix", "effective_checkpoint_time_proxy"):
                section = result[label]
                print(f"\n{label}: {section['total_user_turns']:,} User turns")
                for word in words:
                    print(f"  {word:<18} {section['selected'][word]:>9,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
