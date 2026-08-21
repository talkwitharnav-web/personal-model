"""Why does "hey!" get answered with "You're welcome, man"?

Router does its job - `chitchat 1.00`, no retrieval - so this is the model, and it is worth
understanding because a greeting is the most common first thing anyone types.

The hypothesis to test: the conversational sets (SODA 40k, PersonaChat 8.6k, OASST1 16.8k) are
almost entirely MULTI-TURN. When packing builds a training row it keeps the whole conversation, so
the assistant turns the model actually learns from are mostly deep inside a conversation with lots
of prior context. A bare opening greeting with NOTHING before it may be very rare - in which case
the model has no idea what an opening turn looks like and falls back to the register it knows,
which is mid-conversation reply.

Two measurements:
  1. how the model behaves on short openers, sampled several times (one sample proves nothing)
  2. how often the training data actually contains an opening greeting as the FIRST turn
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OPENERS = ["hey!", "hey", "hi", "hello", "yo", "good morning", "what's up",
           "hey there", "hi there", "howdy"]

GREETING_RE = re.compile(r"^(hey|hi|hello|yo|howdy|good (morning|afternoon|evening)|"
                         r"what'?s up|sup)\b", re.I)

CONVERSATIONAL = ["soda", "personachat", "oasst1", "no_robots", "dailydialog"]


def measure_data():
    print("=" * 96)
    print("TRAINING DATA: how often is the FIRST user turn a bare greeting?")
    print("=" * 96)
    bare_openers = Counter()
    for name in CONVERSATIONAL:
        path = Path("data") / name / "train.txt"
        if not path.exists():
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read(60_000_000)
        blocks = [b.strip() for b in text.split("<|endoftext|>") if b.strip()]
        first_is_greeting = 0
        short_opener = 0
        examined = 0
        for block in blocks:
            lines = block.split("\n")
            user_lines = [l for l in lines if l.startswith("User:")]
            if not user_lines:
                continue
            examined += 1
            body = user_lines[0][len("User:"):].strip()
            if GREETING_RE.match(body):
                first_is_greeting += 1
                # A BARE greeting is the whole turn - "hey!" - as opposed to a greeting that
                # opens a real message, "Hey Shavon, what's up? You seem troubled.". Only the
                # first teaches the model what to do when a user just says hello.
                if len(body.split()) <= 2:
                    bare_openers[body.lower().strip("!?., ")] += 1
            if len(body.split()) <= 3:
                short_opener += 1
        if not examined:
            continue
        print(f"  {name:<14} {examined:>7,} conversations | "
              f"opens with a greeting {first_is_greeting:>6,} ({100*first_is_greeting/examined:4.1f}%) | "
              f"opener <=3 words {short_opener:>6,} ({100*short_opener/examined:4.1f}%)")

    print("\n  BARE greetings (the whole first turn, <=2 words) across all of the above:")
    total_bare = sum(bare_openers.values())
    if not total_bare:
        print("      NONE - the model has never seen a conversation that opens with just a greeting")
    for text, count in bare_openers.most_common(12):
        print(f"      {text!r:<18} {count:>6,}")
    print(f"      total {total_bare:,}")


def measure_model(checkpoint, device, samples):
    from tools.rag_vs_bare_peek import load, generate
    model, tokenizer, step, best = load(checkpoint, device)
    print(f"\n{'=' * 96}")
    print(f"MODEL: {checkpoint} (step {step:,}) - {samples} samples per opener")
    print("=" * 96)
    for opener in OPENERS:
        prompt = f"User: {opener}\nAssistant:"
        answers = Counter()
        for seed in range(samples):
            answers[generate(model, tokenizer, prompt, device, 28,
                             greedy=False, seed=seed)[:70]] += 1
        greedy = generate(model, tokenizer, prompt, device, 28, greedy=True)
        print(f"\n  '{opener}'")
        print(f"      greedy   {greedy[:84]}")
        for text, count in answers.most_common(2):
            print(f"      x{count:<2}      {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/finetune_v2_best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()

    measure_data()
    if not args.data_only:
        measure_model(args.checkpoint, args.device, args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
