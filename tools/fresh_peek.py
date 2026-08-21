"""A conversational peek with questions the model has never been tested on.

Everything in `tests/behaviour_regression_v2.py` and `tools/wide_peek.py` has now been asked so many
times that it is effectively a memorised exam. This file exists to ask NEW things, and to test two
shapes the other tools never touch:

  multi-turn   does the model hold context across turns, or answer each line in isolation? The mix
               carries SODA (40k), PersonaChat (8.6k) and OASST1 (16.8k) precisely for this.
  follow-up    a pronoun in the second turn ("what colour is it?") can only be resolved by reading
               the first. This is the cheapest real test of whether attention is doing its job
               across a conversation rather than pattern-matching a single line.

Greedy and sampled are both printed, because one sample cannot tell "got worse" from "rolled badly"
(the lesson from `tools/peek_variance.py`).
"""
import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tools.rag_vs_bare_peek import load, generate

# Single-turn questions never used in any other harness.
FRESH = [
    ("everyday", "Why do people wear coats in winter?"),
    ("everyday", "What do bees make?"),
    ("everyday", "Why should you look both ways before crossing the road?"),
    ("reasoning", "If it is raining, should you take an umbrella?"),
    ("reasoning", "Tom has three apples and eats one. How many are left?"),
    ("reasoning", "Which is heavier, a kilogram of feathers or a kilogram of stone?"),
    ("advice", "How do I make a cup of tea?"),
    ("advice", "What should I do if I cannot sleep?"),
    ("opinion", "What makes a good friend?"),
    ("creative", "Tell me a short story about a lost cat."),
    ("creative", "Describe the sea in one sentence."),
    ("refusal", "What is the exact population of my street?"),
]

# Each conversation is a list of user turns delivered one at a time; the model's own replies are
# fed back in, so a failure to carry context shows up immediately.
CONVERSATIONS = [
    ["Hi! My name is Sam.", "What is my name?"],
    ["I have a dog called Rex.", "What colour is he?", "Is he a good dog?"],
    ["I like pizza.", "What should I eat for dinner?"],
]


def wrap(label, text):
    print(f"      {label:<9} " + textwrap.fill(text or "(empty)", width=94,
                                               subsequent_indent=" " * 16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/finetune_v2_best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=44)
    args = parser.parse_args()

    model, tokenizer, step, best = load(args.checkpoint, args.device)
    print(f"checkpoint {args.checkpoint} (step {step:,}, validation {best:.4f})")

    current = None
    for group, question in FRESH:
        if group != current:
            current = group
            print(f"\n{'=' * 100}\n{group.upper()}\n{'=' * 100}")
        print(f"\n  Q: {question}")
        prompt = f"User: {question}\nAssistant:"
        wrap("greedy", generate(model, tokenizer, prompt, args.device, args.tokens, greedy=True))
        wrap("sampled", generate(model, tokenizer, prompt, args.device, args.tokens,
                                 greedy=False, seed=3))

    print(f"\n{'=' * 100}\nMULTI-TURN (does it carry context across turns?)\n{'=' * 100}")
    for turns in CONVERSATIONS:
        print()
        history = ""
        for turn in turns:
            history += f"User: {turn}\nAssistant:"
            reply = generate(model, tokenizer, history, args.device, args.tokens, greedy=True)
            print(f"      User      {turn}")
            wrap("Assistant", reply)
            history += f" {reply}\n"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
