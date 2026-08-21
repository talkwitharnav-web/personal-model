"""Run the REAL product - router, retriever and model - on the evaluation questions.

Everything measured today called the model directly. That answers "how good is the model?" but not
"how good is the thing Dark actually ships", and those differ enormously because the model is
92.8% with a passage and 57.0% without one. A bare-prompt test puts every factual question on the
57% path, when in production the router sends them to the 92.8% path.

This runs `orchestrator.pipeline.Pipeline`, so each question takes whichever path the router
chooses, with real passages retrieved from the real Wikipedia index.
"""
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from orchestrator.pipeline import Assistant

QUESTIONS = [
    ("knowledge", "What is the capital of France?"),
    ("knowledge", "Who invented the Tesla coil?"),
    ("knowledge", "When did the Second World War begin?"),
    ("knowledge", "What is photosynthesis?"),
    ("knowledge", "What is a mitochondrion?"),
    ("knowledge", "How many legs does a cat have?"),
    ("knowledge", "What colour is the sky on a clear day?"),
    ("knowledge", "What do bees make?"),
    ("chitchat", "Hi! My name is Sam."),
    ("chitchat", "What is your favourite colour?"),
    ("chitchat", "thats so funny lmao"),
    ("chitchat", "Thank you very much!"),
    ("maths", "Tom has three apples and eats one. How many are left?"),
    ("messy", "wat is photosynthesis"),
    ("messy", "can u help me"),
    ("messy", "What is teh capitol of Frnace?"),
]


def main():
    pipeline = Assistant(device="cuda", quiet=True)
    print()
    current = None
    for group, question in QUESTIONS:
        if group != current:
            current = group
            print(f"\n{'=' * 98}\n{group.upper()}\n{'=' * 98}")
        route = pipeline.plan(question)
        answer = pipeline.answer(question, route)
        path = "RAG" if route.used_retrieval else "no context"
        print(f"\n  Q  {question}")
        print(f"     [{route.category} {route.confidence:.2f} -> {path}"
              + (f", {len(route.passages)} passages]" if route.used_retrieval else "]"))
        print(textwrap.fill(answer or "(empty)", 94,
                            initial_indent="     A  ", subsequent_indent="        "))
    pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


