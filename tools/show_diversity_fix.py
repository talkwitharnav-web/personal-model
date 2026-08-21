"""Show the phrasing-diversity fix: old generated questions against new ones.

The lesson that drove this: a fact learned at one phrasing is not a fact, it is a memorised
string. The pretrained model knows "Paris is the capital and most populous city of France" -
Wikipedia's exact opening formula - but answers "the province of Quebec, Canada" to "Paris is
the capital".

definitions had ELEVEN question templates and wrote every question in tidy sentence case, so it
taught the skill at eleven shapes. This prints the same kind of question before and after so the
difference is readable rather than a statistic.
"""
import random
import sys
from collections import Counter
from pathlib import Path

from tools.audit_finetune_datasets import read_records

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OLD_DIR = Path("C:/Users/arnav/.copilot/session-state/"
               "12f79a3d-2c7f-441b-b5a2-5cbbfea6f4d7/files")
PAIRS = [
    ("definitions", OLD_DIR / "definitions_OLD.txt", Path("data/definitions/train.txt")),
    ("gsm8k_clean", OLD_DIR / "gsm8k_OLD.txt", Path("data/gsm8k_clean/train.txt")),
]
SHOW = 12


def questions_of(path, limit=4000):
    if not path.exists():
        return []
    return [r["question"] for r in read_records(path, limit) if r["question"]]


def answers_of(path, limit=4000):
    if not path.exists():
        return []
    return [r["answer"] for r in read_records(path, limit) if r["answer"]]


def main():
    rng = random.Random(7)

    for name, old_path, new_path in PAIRS:
        print("=" * 84)
        print(f"{name}")
        print("=" * 84)

        old_q = questions_of(old_path)
        new_q = questions_of(new_path)
        if not old_q or not new_q:
            print("  (missing one side, skipping)\n")
            continue

        if name == "definitions":
            print("\n  BEFORE - 11 templates, every question in tidy sentence case")
            for question in rng.sample(old_q, min(SHOW, len(old_q))):
                print(f"    {question[:96]}")
            print("\n  AFTER - 48 templates plus lowercase and unpunctuated styles")
            for question in rng.sample(new_q, min(SHOW, len(new_q))):
                print(f"    {question[:96]}")

            for label, questions in (("before", old_q), ("after", new_q)):
                openings = Counter(" ".join(q.lower().split()[:2]) for q in questions)
                top_ten = sum(c for _, c in openings.most_common(10)) / len(questions)
                lower = sum(1 for q in questions if q[:1].islower()) / len(questions)
                no_mark = sum(1 for q in questions
                              if not q.rstrip().endswith("?")) / len(questions)
                print(f"\n  {label:<6} top-10 openings cover {top_ten*100:5.1f}% | "
                      f"lowercase {lower*100:4.1f}% | no question mark {no_mark*100:4.1f}%")
            print("  human reference: squad 36.9% / sciq 33.7% / no_robots 44.0%")

        else:
            old_a = answers_of(old_path)
            new_a = answers_of(new_path)
            print("\n  BEFORE - 5 closing phrases")
            for answer in rng.sample(old_a, min(6, len(old_a))):
                print(f"    ...{answer.strip().splitlines()[-1][:80]}")
            print("\n  AFTER - 18 closing phrases")
            for answer in rng.sample(new_a, min(6, len(new_a))):
                print(f"    ...{answer.strip().splitlines()[-1][:80]}")
            for label, answers in (("before", old_a), ("after", new_a)):
                endings = Counter(a.strip().splitlines()[-1].split()[0].lower()
                                  for a in answers if a.strip())
                top = endings.most_common(1)[0]
                print(f"\n  {label:<6} distinct closing first-words {len(endings)} | "
                      f"most common {top[0]!r} at {top[1]/len(answers)*100:.1f}%")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
