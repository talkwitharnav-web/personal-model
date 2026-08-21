"""5% replay - raw pretraining passages mixed INTO the fine-tune set.

Section 10: replay lives IN the combined dataset, not in a separate stage, so old knowledge is
actively reinforced rather than passively preserved by a low learning rate. A separate stage is
what produced the SQuAD v2 collapse (section 8).

⚠ THE TRAP, quoted from section 10: "replay rows have no `\\nAssistant:` marker, so every tokenize
script would silently discard them (`if answer_start is None: return None`). They need the
opposite treatment - no masking at all, train on every token. Needs a flag."

That flag is NO_MASK_DATASETS in data/pack_finetune_datasets.py, and "replay" is in it. Without
it this dataset would vanish at pack time and nothing would say so - the row count would simply
be lower than expected.

Passages are drawn from the MIXED pretrain corpus (data/pretrain/pretrain_train.txt), so replay
reflects the same 80/15/5 Wikipedia/TinyStories/WikiText blend the model was pretrained on rather
than Wikipedia alone.
"""
import sys
from pathlib import Path

from data.clean_wikipedia import iter_articles

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SOURCE = Path("data/pretrain/pretrain_train.txt")
OUT_DIR = Path("data/replay")

# 5% of the fine-tune mix. The packed train set is ~167,700 rows of 768 tokens, so 5% is about
# 6.4M tokens; at ~2,400 characters (~600 tokens) per passage that is ~10,600 passages.
TARGET = {"train": 10_600, "validation": 260}
MIN_CHARS = 400
MAX_CHARS = 2_400          # stays inside the 768-token window after tokenizing
# Skip far into the corpus so replay passages are not the same text the model saw first, and
# stride so they are spread across it rather than clustered.
START_SKIP = 40_000
STRIDE = 37


def build(split_name, target):
    if not SOURCE.exists():
        raise SystemExit(f"missing {SOURCE} - run data/build_pretrain_corpus.py first")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    seen = 0
    offset = START_SKIP if split_name == "train" else 5

    with open(OUT_DIR / f"{split_name}.txt", "w", encoding="utf-8") as handle:
        for document in iter_articles(str(SOURCE)):
            seen += 1
            if seen < offset or (seen - offset) % STRIDE:
                continue
            # Take the opening of the document, cut on a sentence boundary so passages do not
            # end mid-clause - training on truncated text teaches trailing off.
            text = " ".join(document.split())
            if len(text) < MIN_CHARS:
                continue
            if len(text) > MAX_CHARS:
                cut = text.rfind(". ", 0, MAX_CHARS)
                text = text[:cut + 1] if cut > MIN_CHARS else text[:MAX_CHARS]

            handle.write(f"{text}\n<|endoftext|>\n")
            written += 1
            if written >= target:
                break
            if written % 5_000 == 0:
                print(f"    {written:,} written")

    print(f"  {split_name}: {written:,} passages from {seen:,} documents scanned")
    return written


def main():
    print("preparing 5% replay from the mixed pretrain corpus")
    build("train", TARGET["train"])
    build("validation", TARGET["validation"])

    sample = (OUT_DIR / "train.txt").read_text(encoding="utf-8")
    first = sample.split("<|endoftext|>")[0].strip()
    print(f"\nfirst passage ({len(first)} chars):")
    print("  " + first[:400])
    if "\nAssistant:" in sample:
        raise SystemExit("FAIL: replay must NOT contain an Assistant marker - it is raw text")
    print("\nverified: no Assistant marker (correct - these rows train on every token, "
          "via NO_MASK_DATASETS)")


if __name__ == "__main__":
    main()
