"""ProofWriter, regrouped so "always answer Unknown" cannot win.

THE DEFECT (found by tools/audit_finetune_datasets.py):
The original script shuffles 585k rows and takes 80,000, so each theory (the fact/rule set) lands
with roughly ONE question. Answers came out Unknown 45.7% / True 27.3% / False 27.0%, which means
a model that ignores the statement entirely and always answers "Unknown" scores 45.7% on the
second-largest dataset in the fine-tune mix - and nothing in the data punishes it.

This is the BoolQ defect wearing a different hat. Onboarding section 9: every BoolQ passage
appeared with exactly one question, so "echo the passage" scored 100% and the loss never
objected. A balance audit alone would not catch it - and indeed the original mix is not wildly
imbalanced, it is just imbalanced enough for one label to be a good bet.

THE FIX, same principle that worked for the contrastive yes/no set:
Keep several questions PER THEORY, deliberately chosen to have DIFFERENT answers. Then no single
answer can satisfy a theory's group, and the only way to score is to read the statement. A
positional or majority rule is guaranteed to lose points on every group it touches.

Also balances the three classes, the way BoolQ was downsampled to 3,553/3,553.
"""
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

from tools.memtrack import report

OUTPUT_DIR = Path("data/proofwriter")

# ProofWriter is the single biggest source of the verdict-word bias measured on 2026-08-19.
# In the packed training data the FIRST token of an answer decides the whole reply, and that
# position looked like this:
#
#     ' a' 5.53% | ' It' 4.39% | ' True' 4.00% | ' Unknown' 3.91% | ' Fal'(se) 3.87%
#     ' No' 3.84% | ' Yes' 3.84%
#
# True + False + Unknown alone are 11.8%, all of it from this file's 20,000 rows. One reply in
# five across the whole mix opens with a bare logical verdict, so an uncertain model reaches for
# one - which is why "what does hot mean?" was answered "Yes" and "are fishes and cats the same?"
# was answered "Yes".
#
# The label has to stay - it is what makes the task checkable - so instead the SURFACE FORM varies.
# The verdict is still there and still first in most variants, but it is no longer the same single
# token 20,000 times, and a third of the rows now open with something else entirely.
#
# ⚠ The first version of this list included "Yes, that's true." and "No, that's false.", which
# traded one bias for another: True/False/Unknown fell from 11.8% to 4.3% of answer-openings while
# Yes/No ROSE from 7.7% to 9.6%, because ProofWriter is large enough that whatever it opens with
# becomes a mix-wide prior. Yes/No are already the second-biggest source of the verdict habit, so
# no phrasing here may start with either.
VERDICT_PHRASINGS = {
    "True": ["True", "True.", "That's true.", "Correct.", "That is correct.",
             "That follows from the statements.", "True - it follows from the statements.",
             "It follows."],
    "False": ["False", "False.", "That's false.", "Incorrect.", "That is not correct.",
              "That does not follow.", "False - it contradicts the statements.",
              "It contradicts the statements."],
    "Unknown": ["Unknown", "Unknown.", "That's unknown.", "There isn't enough information.",
                "It cannot be determined from the statements.", "Not enough information to say.",
                "Unknown - the statements don't cover it.", "There's no way to tell."],
}


def phrase_answer(answer, rng):
    """One of several surface forms for the same verdict."""
    return rng.choice(VERDICT_PHRASINGS.get(answer, [answer]))
TARGET = {"train": 80_000, "validation": 5_000}

# Bounded so the grouping dict cannot grow with the 585k-row source. Each theory keeps several
# questions, so this many theories is ample to reach the row target.
MAX_THEORIES = {"train": 40_000, "validation": 3_000}
QUESTIONS_PER_THEORY = 3
SEED = 42


def collect(split, max_theories):
    """Group questions by theory, holding only a bounded number of theories."""
    grouped = defaultdict(list)
    skipped = 0
    for example in split:
        theory = example["theory"].strip()
        question = example["question"].strip()
        answer = example["answer"].strip()
        if not theory or not question or not answer:
            continue
        if theory not in grouped and len(grouped) >= max_theories:
            skipped += 1
            continue
        grouped[theory].append((question, answer))
    return grouped, skipped


def choose(questions, rng):
    """Pick up to QUESTIONS_PER_THEORY questions covering as many distinct answers as possible.

    Rarest class FIRST. Ordering by the largest class instead pushes the majority label into
    every group twice and barely moves the distribution - measured at 42.8% against the original
    45.7%, which is not a fix. Taking one of each answer in ascending order of availability gives
    a theory whose questions have three different answers, so no single guess can score above
    chance on it.
    """
    by_answer = defaultdict(list)
    for question, answer in questions:
        by_answer[answer].append(question)
    for bucket in by_answer.values():
        rng.shuffle(bucket)

    order = sorted(by_answer, key=lambda a: len(by_answer[a]))
    chosen = []
    while len(chosen) < QUESTIONS_PER_THEORY:
        progressed = False
        for answer in order:
            if by_answer[answer]:
                chosen.append((by_answer[answer].pop(), answer))
                progressed = True
                if len(chosen) >= QUESTIONS_PER_THEORY:
                    break
        if not progressed:
            break
    return chosen


def balance(rows, target, rng):
    """Downsample to equal answer classes, the way BoolQ was cut to 3,553/3,553.

    Per-theory selection alone cannot guarantee balance, because theories that lack a class
    contribute skewed groups. An explicit global cut does guarantee it, so 'always guess X'
    is worth exactly chance.
    """
    by_answer = defaultdict(list)
    for row in rows:
        by_answer[row[2]].append(row)
    smallest = min(len(v) for v in by_answer.values())
    per_class = min(smallest, target // len(by_answer))
    balanced = []
    for bucket in by_answer.values():
        rng.shuffle(bucket)
        balanced.extend(bucket[:per_class])
    rng.shuffle(balanced)
    return balanced


def build(split_name, split, target, max_theories):
    rng = random.Random(SEED)
    grouped, skipped = collect(split, max_theories)
    print(f"  {len(grouped):,} theories kept ({skipped:,} rows skipped past the cap)")
    sizes = Counter(len(v) for v in grouped.values())
    print(f"  questions per theory: min {min(sizes)} | max {max(sizes)}")

    rows = []
    multi = 0
    for theory, questions in grouped.items():
        chosen = choose(questions, rng)
        if len({a for _, a in chosen}) > 1:
            multi += 1
        for question, answer in chosen:
            rows.append((theory, question, answer))

    before = Counter(a for _, _, a in rows)
    rows = balance(rows, target, rng)
    after = Counter(a for _, _, a in rows)
    print(f"  before balancing: " + ", ".join(f"{a} {c}" for a, c in before.most_common()))

    output = OUTPUT_DIR / f"{split_name}.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phrasing = random.Random(31)
    with open(output, "w", encoding="utf-8") as handle:
        for theory, question, answer in rows:
            handle.write(f"User: {theory}\n{question}\n"
                         f"Assistant: {phrase_answer(answer, phrasing)}\n<|endoftext|>\n")

    answers = Counter(a for _, _, a in rows)
    theories = len({t for t, _, _ in rows})
    total = len(rows)
    print(f"  wrote {output}: {total:,} rows over {theories:,} theories "
          f"({total/max(theories,1):.2f} questions per theory)")
    spread = ", ".join(f"{a} {c/total*100:.1f}%" for a, c in answers.most_common())
    print(f"  answers: {spread}")
    print(f"  theories whose questions have MORE THAN ONE answer: {multi:,} "
          f"- a single guess loses points on every one of them")
    return answers, total


def main():
    print("loading tasksource/proofwriter ...")
    dataset = load_dataset("tasksource/proofwriter")
    for split_name in ("train", "validation"):
        print(f"\n=== {split_name} ===")
        build(split_name, dataset[split_name], TARGET[split_name], MAX_THEORIES[split_name])
    report("(end)")


if __name__ == "__main__":
    main()
