"""How badly do typos and internet shorthand break the model, and why?

Two questions, and the second explains the first.

WHY IT BREAKS - the tokenizer. `tokenizer_v2` is BPE fitted on 2 GB of cleaned Wikipedia. A word
that appears constantly there gets its own single token; a misspelling appears nowhere, so it is
shattered into fragments. "like" and "loke" therefore do not live near each other in any sense the
model can use - they are different token ids with independently learned embeddings, and nothing in
training ever told the model they mean the same thing. The same is true of "lmao", "idk" and "tbh",
which Wikipedia essentially never writes.

This is not fixable after the fact by prompting. The fix is either data (train on corrupted copies
of the same sentences so the fragments learn to point at the same meaning) or serving-side
normalisation (spellcheck before the model sees it). This file measures the size of the problem so
that choice can be made on evidence.

Prints, for each pair:
  - how the tokenizer splits the clean and the corrupted word
  - what the model answers to each
"""
import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tools.rag_vs_bare_peek import load, generate

# (clean, corrupted) - the corruption is the sort a real person makes: an adjacent key, a
# transposition, a doubled or dropped letter.
TYPO_PAIRS = [
    ("I like apples.", "I loke apples."),
    ("What is the capital of France?", "What is teh capitol of Frnace?"),
    ("Can you help me?", "can u help me"),
    ("Tell me about dogs.", "tel me abut dogs"),
    ("That is very funny.", "thats so funny lmao"),
    ("I do not know the answer.", "idk the answer tbh"),
    ("Are you there?", "r u there"),
    ("Thank you very much!", "ty so much!!"),
]

WORDS = ["like", "loke", "the", "teh", "capital", "capitol", "you", "u", "your", "ur",
         "laughing", "lmao", "know", "idk", "honestly", "tbh", "please", "pls"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/finetune_v2_best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=36)
    args = parser.parse_args()

    model, tokenizer, step, best = load(args.checkpoint, args.device)
    print(f"checkpoint {args.checkpoint} (step {step:,}, validation {best:.4f})")

    print(f"\n{'=' * 96}\nHOW THE TOKENIZER SEES THESE WORDS\n{'=' * 96}")
    print(f"  {'word':<12}{'tokens':<8}{'pieces'}")
    print("  " + "-" * 78)
    for word in WORDS:
        encoded = tokenizer.encode(" " + word)
        pieces = [tokenizer.decode([i]) for i in encoded.ids]
        print(f"  {word:<12}{len(encoded.ids):<8}{pieces}")
    print("\n  A word split into fragments has no shared embedding with its correct spelling.")
    print("  Nothing in training ever taught the model that the fragments mean the same thing.")

    print(f"\n{'=' * 96}\nWHAT THE MODEL DOES WITH EACH\n{'=' * 96}")
    for clean, messy in TYPO_PAIRS:
        print(f"\n  clean  {clean}")
        answer = generate(model, tokenizer, f"User: {clean}\nAssistant:",
                          args.device, args.tokens, greedy=True)
        print(textwrap.fill(answer or "(empty)", 92, initial_indent="      -> ",
                            subsequent_indent="         "))
        print(f"  messy  {messy}")
        answer = generate(model, tokenizer, f"User: {messy}\nAssistant:",
                          args.device, args.tokens, greedy=True)
        print(textwrap.fill(answer or "(empty)", 92, initial_indent="      -> ",
                            subsequent_indent="         "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
