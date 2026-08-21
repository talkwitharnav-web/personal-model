"""Print readable samples from the conversational datasets in the v2 mix.

Kept as a file rather than an inline snippet because the raw text contains quotes, dashes and
newlines that PowerShell here-strings mangle.

The point of comparison is v1's DailyDialog, which the v2 mix drops. DailyDialog is short,
clipped, ESL-textbook dialogue; SODA, PersonaChat, No Robots and OASST1 are full-length modern
conversation and instruction data. The register the model writes in is inherited almost directly
from these files.
"""
import argparse
import sys
import textwrap
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def show(name, count, maxlen, skip):
    path = Path("data") / name / "train.txt"
    if not path.exists():
        print(f"--- {name}: no train.txt")
        return
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read(500_000)
    blocks = [b.strip() for b in text.split("<|endoftext|>") if b.strip()]
    print("=" * 98)
    print(f"{name.upper()}")
    print("=" * 98)
    for block in blocks[skip:skip + count]:
        body = block if len(block) <= maxlen else block[:maxlen] + "  ...[cut]"
        for line in body.split("\n"):
            print(textwrap.fill(line, 96, subsequent_indent="            ") if line.strip() else "")
        print("-" * 98)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="no_robots,soda,personachat,oasst1,dailydialog")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--maxlen", type=int, default=1000)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()
    for name in args.datasets.split(","):
        show(name, args.count, args.maxlen, args.skip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
