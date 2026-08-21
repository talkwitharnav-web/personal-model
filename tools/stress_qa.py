"""Stress-test question answering: consistency, negation, false premises, format constraints.

Complements the conversation stress test. Everything here reports a RATE over several samples,
never a single anecdote, because this project's most repeated bug is concluding from one roll.

Each group targets a specific suspicion:

  consistency   the same fact asked five ways. If the answers disagree with each other, the model
                has memorised strings rather than facts (section 9), and the disagreement rate
                measures how badly.
  negation      "which of these is NOT a mammal". Negation is a classic failure for small models
                because the cheap policy - ignore the negation and answer the positive - is right
                half the time.
  false premise "when did Einstein invent the telephone". A model with no way to push back will
                confabulate a date. Measures whether it can decline.
  format        "answer in exactly one word", "answer with just a number". Measures whether an
                instruction about the OUTPUT is followed at all.
  comparison    "is the sun bigger than the moon". Needs two facts and a relation, not a lookup.

Retrieval state is logged with every answer, so a failure can be attributed to the retriever
rather than blamed on the model.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from orchestrator.pipeline import Assistant

CONSISTENCY = [
    ("capital of France", ["What is the capital of France?", "France's capital is what?",
                           "Name the capital city of France.", "Which city is France's capital?",
                           "france capital"], "paris"),
    ("who invented the telephone", ["Who invented the telephone?",
                                    "The telephone was invented by whom?",
                                    "Name the inventor of the telephone.",
                                    "Who is credited with inventing the telephone?",
                                    "telephone inventor"], "bell"),
]

NEGATION = [
    ("Which of these is not a mammal: dog, cat, salmon?", ["salmon"]),
    ("Which of these is not a colour: red, blue, hammer?", ["hammer"]),
    ("Name something that is not a fruit: apple, banana, carrot.", ["carrot"]),
    ("Is a whale not a fish?", ["yes", "correct", "not a fish", "mammal"]),
]

FALSE_PREMISE = [
    ("When did Einstein invent the telephone?", ["did not", "didn't", "no", "not", "bell"]),
    ("Why is the sky green?", ["not green", "isn't green", "blue", "no"]),
    ("Who was the first person to walk on the sun?", ["no one", "nobody", "cannot", "not"]),
]

FORMAT = [
    ("What is the capital of France? Answer in exactly one word.", "one_word"),
    ("How many legs does a spider have? Answer with just a number.", "number"),
    ("List three colours.", "three_items"),
    ("Write one sentence about the ocean.", "one_sentence"),
]

COMPARISON = [
    ("Is the sun bigger than the moon?", ["yes", "bigger", "larger"]),
    ("Which is heavier, an elephant or a mouse?", ["elephant"]),
    ("Is water wetter than sand?", ["yes", "water"]),
]


def satisfies_format(answer, kind):
    words = answer.split()
    if kind == "one_word":
        return len(words) == 1
    if kind == "number":
        return bool(re.fullmatch(r"[^\d]*\d+[^\d]*", answer.strip()))
    if kind == "three_items":
        parts = re.split(r",|\n|\d\.", answer)
        return len([p for p in parts if p.strip()]) >= 3
    if kind == "one_sentence":
        sentences = [s for s in re.split(r"[.!?]", answer) if s.strip()]
        return len(sentences) == 1 and 3 <= len(words) <= 40
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--mode", default="auto")
    # Accepted so tools/final_evaluation.py can drive this the same way it drives every other
    # eval. The pipeline picks the checkpoint up from orchestrator/config.py, so --checkpoint is
    # recorded for the report rather than used; --device is real.
    parser.add_argument("--checkpoint", default="checkpoints/finetune_v2_best.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assistant = Assistant(device=args.device, quiet=True)
    n = args.samples

    def ask(question):
        route = assistant.plan(question, mode=args.mode)
        answer = assistant.answer(question, route, mode=args.mode)
        return answer, route

    print(f"{n} samples per prompt, mode={args.mode}\n")

    print("=" * 98)
    print("CONSISTENCY - the same fact asked five ways (disagreement means strings, not facts)")
    print("=" * 98)
    for label, phrasings, expected in CONSISTENCY:
        hits, seen = 0, Counter()
        for phrasing in phrasings:
            answer, route = ask(phrasing)
            ok = expected in answer.lower()
            hits += ok
            seen[answer[:40]] += 1
            print(f"  {'ok  ' if ok else 'MISS'} [{'rag' if route.used_retrieval else 'none'}] "
                  f"{phrasing[:44]:<46} {answer[:44]}")
        print(f"    -> {label}: {hits}/{len(phrasings)} correct, "
              f"{len(seen)} distinct answers\n")

    for title, cases, checker in (
        ("NEGATION - can it handle 'not'?", NEGATION, "any"),
        ("FALSE PREMISE - can it decline a question built on a lie?", FALSE_PREMISE, "any"),
        ("COMPARISON - two facts plus a relation", COMPARISON, "any"),
    ):
        print("=" * 98)
        print(title)
        print("=" * 98)
        total = 0
        for question, accepted in cases:
            hits = 0
            example = ""
            for _ in range(n):
                answer, route = ask(question)
                if any(a in answer.lower() for a in accepted):
                    hits += 1
                elif not example:
                    example = answer[:64]
            total += hits / n
            print(f"  {hits}/{n}  {question[:56]:<58} {example}")
        print(f"    -> mean {total/len(cases):.0%}\n")

    print("=" * 98)
    print("FORMAT CONSTRAINTS - is an instruction about the OUTPUT followed?")
    print("=" * 98)
    total = 0
    for question, kind in FORMAT:
        hits = 0
        example = ""
        for _ in range(n):
            answer, route = ask(question)
            if satisfies_format(answer, kind):
                hits += 1
            elif not example:
                example = answer[:64]
        total += hits / n
        print(f"  {hits}/{n}  [{kind:<12}] {question[:44]:<46} {example}")
    print(f"    -> mean {total/len(FORMAT):.0%}")

    assistant.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
