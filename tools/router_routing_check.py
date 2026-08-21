"""Does the ROUTER send each test question down the right path?

Every evaluation run today - `tests/behaviour_FINAL.txt`, `tools/rag_vs_bare_peek.py`,
`tools/fresh_peek.py` - called the model DIRECTLY. None of them went through
`orchestrator/pipeline.py`, so none of them exercised the router that decides whether a question
gets a retrieved passage.

That matters because the model's two capabilities are wildly different, measured on held-out data:

    with a passage      92.8%
    without a passage   57.0%   (chance is 50%)

So "is this question routed to retrieval?" is very nearly the whole story for whether an answer
will be right. A bare-prompt evaluation is measuring the 57% path for questions that would, in
production, have taken the 92.8% path - which makes the model look far worse than it is.

`NO_RETRIEVAL_CATEGORIES = {"chitchat", "maths", "summarise"}` and retrieval is the default, so a
question only skips retrieval if the router is BOTH confident (>= 0.6) AND picked one of those
three. This checks what actually happens to the questions the evaluations asked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sentence_transformers import SentenceTransformer

from orchestrator import config
from orchestrator.classifier import Classifier

# The exact questions the evaluations asked, grouped by what SHOULD happen to them.
SHOULD_RETRIEVE = [
    "What is the capital of France?",
    "Who invented the Tesla coil?",
    "When did the Second World War begin?",
    "What is photosynthesis?",
    "What is a mitochondrion?",
    "What is an earthquake?",
    "Is the sun hot?",
    "Do fish have lungs?",
    "Was penicillin discovered by Alexander Fleming?",
    "How many legs does a cat have?",
    "What colour is the sky on a clear day?",
    "What do bees make?",
    "Why do people wear coats in winter?",
    "Why should you look both ways before crossing the road?",
]

SHOULD_SKIP = [
    "What is your favourite colour?",
    "Hi! My name is Sam.",
    "I like pizza.",
    "That is very funny.",
    "thats so funny lmao",
    "Thank you very much!",
    "Are you there?",
    "Tom has three apples and eats one. How many are left?",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
    "clips in May. How many clips did Natalia sell altogether in April and May?",
]

MESSY = [
    "What is teh capitol of Frnace?",
    "can u help me",
    "tel me abut dogs",
    "idk the answer tbh",
    "r u there",
    "wat is photosynthesis",
]


def report(title, questions, classifier, expect_retrieval):
    print(f"\n{'=' * 96}\n{title}\n{'=' * 96}")
    right = 0
    for question in questions:
        category, confidence = classifier.classify(question)
        retrieves = classifier.wants_retrieval(category, confidence)
        ok = (retrieves == expect_retrieval) if expect_retrieval is not None else None
        right += bool(ok)
        mark = "    " if ok is None else ("ok  " if ok else "MISS")
        path = "RETRIEVES" if retrieves else "no context"
        reason = "" if confidence >= config.ROUTER_CONFIDENCE else "  (low confidence)"
        print(f"  {mark} {path:<11} {category:<12} {confidence:.2f}{reason}  {question[:58]}")
    if expect_retrieval is not None:
        print(f"  -> {right}/{len(questions)} routed as intended")
    return right


def main():
    embedder = SentenceTransformer(config.EMBEDDER_NAME)
    classifier = Classifier(embedder)
    print(f"router confidence threshold {config.ROUTER_CONFIDENCE}")
    print(f"skips retrieval only when confident AND in {sorted(config.NO_RETRIEVAL_CATEGORIES)}")

    a = report("SHOULD RETRIEVE - factual, definition, everyday knowledge "
               "(the model is 57% without a passage)",
               SHOULD_RETRIEVE, classifier, expect_retrieval=True)
    b = report("SHOULD SKIP - chitchat and maths (a passage cannot help)",
               SHOULD_SKIP, classifier, expect_retrieval=False)
    report("MESSY INPUT - does a typo derail the ROUTER as well as the model?",
           MESSY, classifier, expect_retrieval=None)

    total = len(SHOULD_RETRIEVE) + len(SHOULD_SKIP)
    print(f"\n{'=' * 96}\nOVERALL: {a + b}/{total} routed as intended\n{'=' * 96}")
    print("Every evaluation today called the model directly, bypassing this. The numbers in")
    print("tests/*_FINAL.txt therefore describe the model, NOT the product.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
