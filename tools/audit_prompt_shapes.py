"""Why do imperatives, negation and false premises fail? Count them in the training data.

Measured failures from `tools/stress_qa.py` (5 samples each):

    "Name the capital city of France."      -> "Yes"          0/5
    "Name the inventor of the telephone."   -> "Yes"          0/5
    "List three colours."                   -> "Yes"          0/5
    "Which of these is not a mammal?"       -> "Sheep"        0/5
    "When did Einstein invent the telephone?" -> "1876"       0/5

The same facts are answered correctly when asked as questions - "Who invented the telephone?"
gives "Alexander Graham Bell" 3 times out of 3. So the knowledge is present and the PHRASING is
what breaks it. That points at the data: if the mix almost never contains a command, the model has
no idea what to do with one and falls back on the verdict-word prior (19.5% of training answers
open with Yes/No/True/False/Unknown).

This counts three things across every dataset in the mix:

  imperative     user turns that are commands ("Name...", "List...", "Give me...") rather than
                 questions. Distinguished from questions by the opening word, not by punctuation,
                 because plenty of real questions omit the question mark.
  negation       user turns containing "not", "n't", "never", "except", "other than".
  unanswerable   assistant turns that decline - the only way a model learns it is allowed to
                 reject a question's premise instead of confabulating a date.
"""
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MIX = ["commonsenseqa", "sciq", "piqa", "boolq", "hellaswag", "socialiqa", "logiqa", "reclor",
       "squad", "race", "xsum", "oasst1", "personachat", "proofwriter", "definitions", "yesno",
       "gsm8k_clean", "soda", "no_robots", "typo_aug", "greetings"]

IMPERATIVE = re.compile(
    r"^\s*(name|list|give|tell|write|describe|explain|summarise|summarize|provide|show|"
    r"identify|state|mention|suggest|compare|define|outline|draft|compose|create|make)\b", re.I)
QUESTION = re.compile(r"^\s*(what|who|when|where|why|how|which|is|are|was|were|do|does|did|"
                      r"can|could|would|should|will|has|have|had)\b", re.I)
NEGATION = re.compile(r"\b(not|never|except|other than|neither|nor)\b|n't\b", re.I)
DECLINE = re.compile(r"\b(i (don't|do not|cannot|can't) know|no one|nobody|there is no|"
                     r"that('s| is) not (true|correct)|isn't (true|correct)|"
                     r"cannot be (answered|determined)|not enough information|"
                     r"the premise|actually,? )", re.I)


def user_turns(block):
    return [line[len("User:"):].strip()
            for line in block.split("\n") if line.startswith("User:")]


def assistant_turns(block):
    return [line[len("Assistant:"):].strip()
            for line in block.split("\n") if line.startswith("Assistant:")]


def main():
    totals = Counter()
    print(f"{'dataset':<15}{'turns':>9}{'imperative':>12}{'question':>11}"
          f"{'negation':>10}{'declines':>10}")
    print("-" * 68)

    for name in MIX:
        path = Path("data") / name / "train.txt"
        if not path.exists():
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read(40_000_000)
        blocks = [b for b in text.split("<|endoftext|>") if b.strip()]

        turns = imperative = question = negation = declines = answers = 0
        for block in blocks:
            for turn in user_turns(block):
                if not turn:
                    continue
                turns += 1
                # First line only: multi-line user turns in SQuAD/RACE carry a passage after the
                # question, and counting the passage's words would swamp the signal.
                head = turn.split("\n")[0]
                if IMPERATIVE.match(head):
                    imperative += 1
                elif QUESTION.match(head):
                    question += 1
                if NEGATION.search(head):
                    negation += 1
            for turn in assistant_turns(block):
                if not turn:
                    continue
                answers += 1
                if DECLINE.search(turn):
                    declines += 1

        if not turns:
            continue
        totals["turns"] += turns
        totals["imperative"] += imperative
        totals["question"] += question
        totals["negation"] += negation
        totals["declines"] += declines
        totals["answers"] += answers
        print(f"{name:<15}{turns:>9,}{100*imperative/turns:>11.1f}%{100*question/turns:>10.1f}%"
              f"{100*negation/turns:>9.1f}%{100*declines/max(answers,1):>9.1f}%")

    t = totals["turns"]
    a = max(totals["answers"], 1)
    print("-" * 68)
    print(f"{'ALL':<15}{t:>9,}{100*totals['imperative']/t:>11.1f}%"
          f"{100*totals['question']/t:>10.1f}%{100*totals['negation']/t:>9.1f}%"
          f"{100*totals['declines']/a:>9.1f}%")

    print("\ninterpretation")
    print("  imperative: a command rather than a question. If this is small, 'Name the capital of")
    print("    France' is a shape the model has barely seen, and it falls back on its prior.")
    print("  declines: the share of ANSWERS that push back instead of complying. If this is near")
    print("    zero the model has never been shown that rejecting a false premise is allowed, so")
    print("    'When did Einstein invent the telephone?' can only produce a date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
