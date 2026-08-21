"""Show what is actually in the fine-tune mix: real counts and a real sample from each dataset.

Reads the source .txt files rather than quoting documented figures - proofwriter and yesno have
both been regenerated, so the numbers in older notes are stale.
"""
import sys
from pathlib import Path

from tools.audit_finetune_datasets import count_records, parse_record

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# The 16 in data/pack_finetune_datasets.py, in the order they appear there.
DATASETS = [
    ("commonsenseqa", "multiple-choice commonsense"),
    ("sciq", "school science QA"),
    ("piqa", "physical commonsense"),
    ("boolq", "yes/no reading comprehension"),
    ("hellaswag", "sentence completion"),
    ("socialiqa", "social reasoning"),
    ("logiqa", "logical reasoning"),
    ("reclor", "logical reading comprehension"),
    ("squad", "extractive QA"),
    ("race", "exam reading comprehension"),
    ("xsum", "single-sentence summarisation"),
    ("oasst1", "open assistant conversations"),
    ("personachat", "persona dialogue"),
    ("proofwriter", "rule-based deduction"),
    ("definitions", "contrastive definitions (built here)"),
    ("yesno", "contrastive yes/no (built here)"),
    ("gsm8k_clean", "grade-school maths word problems"),
    ("soda", "social dialogue (conversational register)"),
    ("no_robots", "human-written instruction following"),
    ("replay", "raw pretrain passages, 5% replay (trains on every token)"),
]

EXCLUDED = [
    ("gsm8k", "grade-school maths, RAW", "superseded by gsm8k_clean - the raw file ends every "
                                         "answer with '#### <number>'"),
    ("dolly", "instruction following", "superseded by OASST1"),
    ("dailydialog", "casual dialogue", "superseded by PersonaChat"),
    ("strategyqa", "multi-hop yes/no", "superseded by the contrastive yes/no set"),
    ("squad_v2", "QA with unanswerables", "the refusal-collapse failure, section 8"),
]

WIDTH = 150


def show(name, description, index=0):
    path = Path(f"data/{name}/train.txt")
    if not path.exists():
        print(f"\n### {name}: MISSING")
        return 0

    total = count_records(path)
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        chunk = handle.read(200_000)
    records = [parse_record(part) for part in chunk.split("<|endoftext|>")]
    records = [r for r in records if r]

    print(f"\n### {name}  -  {description}")
    print(f"    {total:,} examples")
    if not records:
        print("    (could not parse a sample)")
        return total

    record = records[min(index, len(records) - 1)]
    context = " ".join(record["context"].split())
    question = " ".join(record["question"].split())
    answer = " ".join(record["answer"].split())
    if context and context != question:
        print(f"    CONTEXT  {context[:WIDTH]}{'...' if len(context) > WIDTH else ''}")
    print(f"    PROMPT   {question[:WIDTH]}{'...' if len(question) > WIDTH else ''}")
    print(f"    ANSWER   {answer[:WIDTH]}{'...' if len(answer) > WIDTH else ''}")
    return total


def main():
    print("=" * 78)
    print("FINE-TUNE MIX - the 16 datasets packed into data/finetune_packed")
    print("=" * 78)
    grand = 0
    for name, description in DATASETS:
        grand += show(name, description)

    print("\n" + "=" * 78)
    print(f"TOTAL: {grand:,} examples across {len(DATASETS)} datasets")
    print("(packed into 171,702 rows of 768 tokens - 3.7x fewer steps than padded)")

    print("\n" + "=" * 78)
    print("PRESENT ON DISK BUT NOT IN THE MIX")
    print("=" * 78)
    for name, description, reason in EXCLUDED:
        path = Path(f"data/{name}/train.txt")
        count = f"{count_records(path):,}" if path.exists() else "not built"
        print(f"  {name:<14} {count:>10}  {description} - {reason}")


if __name__ == "__main__":
    main()
