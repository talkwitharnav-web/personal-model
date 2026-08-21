"""Measure question-shape diversity per dataset - find template poverty by measuring it.

The project has now hit the same defect three times: a generated dataset teaches a skill only at
the exact phrasings it contains. definitions v1 matched format but not context length. yesno v1
matched format and length but not phrasing, and needed 53 templates. definitions had 11.

Guessing which dataset is next is how it keeps happening. This measures it.

Two signals, both computed on the question only (the part a user varies):

  opening    distribution of the first three words. A generated dataset collapses onto a handful;
             human-written questions spread across thousands.
  coverage   how many distinct openings cover 50% of the questions. Low = template poverty.

Human-written sets (SQuAD, RACE, No Robots) are the reference: they show what real variety looks
like, so a generated set can be compared against something rather than against a guess.
"""
import sys
from collections import Counter
from pathlib import Path

from tools.audit_finetune_datasets import read_records

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GENERATED = ["definitions", "yesno", "gsm8k_clean", "proofwriter"]
HUMAN = ["squad", "race", "no_robots", "oasst1", "sciq"]
SAMPLE = 4000


def opening(question, words=2):
    """First two words: for templated questions this is the template's fingerprint.

    Three words was worse - it captures part of the SUBJECT ("what is a tesla"), so subject
    variety masqueraded as template variety and every dataset looked healthy. Two words keeps
    "what is" / "who made" / "can you" intact.
    """
    parts = question.strip().lower().split()
    return " ".join(parts[:words]) if parts else ""


def measure(name):
    path = Path(f"data/{name}/train.txt")
    if not path.exists():
        return None
    records = read_records(path, SAMPLE)
    if not records:
        return None
    questions = [r["question"] for r in records if r["question"]]
    if not questions:
        return None

    openings = Counter(opening(q) for q in questions)
    total = len(questions)
    top_ten = sum(count for _, count in openings.most_common(10))
    top = openings.most_common(1)[0]
    lowercase = sum(1 for q in questions if q[:1].islower()) / total
    unpunctuated = sum(1 for q in questions if not q.rstrip().endswith("?")) / total

    return {
        "name": name, "questions": total, "distinct": len(openings),
        "top_ten": top_ten / total, "top": top, "top_share": top[1] / total,
        "lowercase": lowercase, "unpunctuated": unpunctuated,
        "examples": [o for o, _ in openings.most_common(5)],
    }


def show(rows, title):
    print(f"\n{title}")
    print(f"  {'dataset':<14} {'distinct':>9} {'top 10':>8} {'top opening':<22} {'share':>7} "
          f"{'lower':>7} {'no ?':>6}")
    print(f"  {'':<14} {'openings':>9} {'cover':>8}")
    for row in rows:
        if not row:
            continue
        print(f"  {row['name']:<14} {row['distinct']:>9,} {row['top_ten']*100:>7.1f}% "
              f"{row['top'][0][:22]:<22} {row['top_share']*100:>6.1f}% "
              f"{row['lowercase']*100:>6.1f}% {row['unpunctuated']*100:>5.1f}%")
    for row in rows:
        if row:
            print(f"    {row['name']:<14} top openings: {', '.join(row['examples'])}")


def main():
    generated = [measure(n) for n in GENERATED]
    human = [measure(n) for n in HUMAN]

    print("=" * 96)
    print("QUESTION-SHAPE DIVERSITY - a high 'top 10 cover' means a handful of templates")
    print("=" * 96)
    show(human, "HUMAN-WRITTEN (the reference for what real variety looks like)")
    show(generated, "GENERATED HERE (ours to fix)")

    print("\ninterpretation")
    print("  'top 10 cover' is the share of all questions starting with one of the ten most")
    print("  common two-word openings. Human sets spread; a templated set concentrates.")
    print("  'lower' and 'no ?' matter too: real users type 'what is a tesla coil', and a set")
    print("  written entirely in tidy prose teaches the model that tidy prose is part of the")
    print("  question - the same defect on a different axis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
