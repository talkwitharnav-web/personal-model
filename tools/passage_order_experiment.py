"""Does the ORDER of retrieved passages change which one the model answers from?

The retrieval fix solved the "capital of France" failure, but two remain and they are not
retrieval failures - the correct passage IS retrieved and ranked first:

    Q: When did the Second World War begin?
       [1] 0.777  'World War II ... The first began ...'          <- correct, ranked first
       [2] 0.762  'Between the First and Second World Wars (1920-1945)'
       -> answered "1920-1945", copied from passage 2

Passages are currently concatenated best-first, which puts the BEST passage FURTHEST from the
position where the answer is generated. In a causal model every generated token attends back over
the whole prompt, but nearby tokens are systematically easier to use - so best-first may be exactly
backwards.

Worth testing rather than assuming, because the training data cuts the other way: both
`prepare_definitions.py` and `prepare_yesno.py` insert the answer-bearing passage at a RANDOM
position, precisely so position could not become the shortcut. If the model really is
position-agnostic, order will make no difference and this experiment says so.

Three orders, same passages, same question:
    best-first   what serving does today
    best-last    highest score nearest the answer
    single       only the top passage, no distractors at all
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from orchestrator.pipeline import Assistant

QUESTIONS = [
    ("When did the Second World War begin?", "1939"),
    ("What is photosynthesis?", "light energy into chemical energy"),
    ("What is the capital of France?", "Paris"),
    ("Who invented the Tesla coil?", "Nikola Tesla"),
    ("What is a mitochondrion?", "organelle"),
    ("How many legs does a cat have?", "four"),
    ("What do bees make?", "honey"),
    ("What colour is the sky on a clear day?", "blue"),
]


def ask(assistant, question, passages):
    prompt, used = assistant.generator.build_prompt(question, passages)
    return "".join(assistant.generator.stream(prompt, temperature=0.2)).strip()


def main():
    assistant = Assistant(device="cuda", quiet=True)
    print("Same passages, three orderings. 'hit' means the expected phrase appears.\n")
    tally = {"best-first": 0, "best-last": 0, "single": 0}

    for question, expected in QUESTIONS:
        route = assistant.plan(question)
        if not route.passages:
            print(f"  (no passages for {question})")
            continue
        orders = {
            "best-first": list(route.passages),
            "best-last": list(reversed(route.passages)),
            "single": route.passages[:1],
        }
        print(f"  Q: {question}   (want: {expected})")
        for label, passages in orders.items():
            answer = ask(assistant, question, passages)
            hit = expected.lower() in answer.lower()
            tally[label] += hit
            print(f"      {'HIT ' if hit else 'miss'} {label:<11} {answer[:88]}")
        print()

    total = len(QUESTIONS)
    print("=" * 92)
    for label, score in tally.items():
        print(f"  {label:<12} {score}/{total}")
    print("=" * 92)
    print("If the three are within one of each other, order does not matter and the remaining")
    print("failures are the model's span selection, not the prompt layout.")
    assistant.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
