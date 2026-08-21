"""Test the ProofWriter regrouping logic without touching the 585k-row dataset.

The whole point of the fix is that a single guessed answer must lose points on most theories.
That property lives in choose(), so it can be tested on synthetic groups - no download, no
memory pressure, and safe to run while a 30-hour training job owns the machine.
"""
import random
import sys
from collections import Counter

from data.prepare_proofwriter_v2 import QUESTIONS_PER_THEORY, balance, choose

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}  {detail}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def main():
    rng = random.Random(0)

    # A theory dominated by Unknown - exactly the shape that produced the 45.7% shortcut.
    skewed = ([(f"q{i}", "Unknown") for i in range(12)]
              + [("qt", "True"), ("qf", "False")])
    chosen = choose(skewed, rng)
    answers = Counter(a for _, a in chosen)
    print(f"skewed theory (12 Unknown / 1 True / 1 False) -> {dict(answers)}")
    check("picks at most QUESTIONS_PER_THEORY", len(chosen) <= QUESTIONS_PER_THEORY,
          f"{len(chosen)}")
    check("recovers more than one answer class from a skewed theory", len(answers) > 1,
          f"{len(answers)} classes")
    check("does not return an all-Unknown group", answers.get("Unknown", 0) < len(chosen),
          f"Unknown {answers.get('Unknown', 0)}/{len(chosen)}")

    # A theory with only one answer available cannot be made contrastive - it must still work.
    single = [(f"q{i}", "True") for i in range(5)]
    chosen = choose(single, rng)
    check("single-class theory still returns questions", len(chosen) > 0, f"{len(chosen)}")
    check("single-class theory returns no duplicates",
          len({q for q, _ in chosen}) == len(chosen))

    # Fewer questions than requested.
    tiny = [("only", "False")]
    chosen = choose(tiny, rng)
    check("theory with one question does not crash", chosen == [("only", "False")], f"{chosen}")

    check("empty theory returns nothing", choose([], rng) == [])

    # The property that matters, measured over many synthetic theories: what does a model that
    # always guesses the majority class actually score?
    rng2 = random.Random(7)
    classes = ["Unknown", "True", "False"]
    rows = []
    for _ in range(4000):
        questions = []
        for i in range(rng2.randint(4, 20)):
            # Deliberately biased source, like the real data.
            answer = rng2.choices(classes, weights=[0.46, 0.27, 0.27])[0]
            questions.append((f"q{i}", answer))
        rows.extend((f"theory", q, a) for q, a in choose(questions, rng2))

    unbalanced = Counter(a for _, _, a in rows)
    total_before = sum(unbalanced.values())
    majority_before = unbalanced.most_common(1)[0]
    print(f"\nover 4,000 synthetic theories -> {total_before:,} rows")
    print(f"  per-theory selection alone: always guessing "
          f"'{majority_before[0]}' scores {majority_before[1]/total_before*100:.1f}%")

    rows = balance(rows, 999_999, rng2)
    distribution = Counter(a for _, _, a in rows)
    total = sum(distribution.values())
    majority = distribution.most_common(1)[0]
    for answer, count in distribution.most_common():
        print(f"    {answer:<8} {count/total*100:5.1f}%")
    print(f"  after global balancing: always guessing '{majority[0]}' scores "
          f"{majority[1]/total*100:.1f}% (was 45.7% in the shipped data, chance is 33.3%)")
    check("majority-guess score is pushed to chance", majority[1] / total < 0.35,
          f"{majority[1]/total*100:.1f}% vs 33.3% chance")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
