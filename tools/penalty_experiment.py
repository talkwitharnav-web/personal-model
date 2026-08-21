"""Does a repetition penalty fix the looping? Measured, at several strengths.

Dark's question: "would some kinda penalty fix the repetition? like for example we could put
penalties on picking a word too often (temperature)".

Two different mechanisms, worth separating:

  temperature         divides the logits before sampling, flattening the distribution. It adds
                      randomness to EVERYTHING; it does not know what a repeat is.
  repetition penalty  divides the score of tokens ALREADY GENERATED, so a token gets harder to
                      pick each time it is used. This is the one that targets looping.

The repo already applies a repetition penalty of 1.3 at serving time (orchestrator/config.py) and
already fixed the important bug in it: it runs over generated tokens only, never the prompt,
because penalising the prompt fines the model for reusing words from the retrieved passage - it
was scoring down "Shakespeare" exactly when Shakespeare was the answer in front of it.

So this measures what a penalty actually buys on a WEAK-domain prompt, where the model loops
because it does not know the topic, versus a STRONG one where it does not loop anyway.
"""
import argparse
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model.config import v2_config
from model.model import DarkGLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOKENIZER = Path("tokenizer/tokenizer_v2.json")
SCRATCH = Path("tests/penalty_checkpoint.tmp")

WEAK = "Quantum entanglement is"
ALSO_WEAK = "The Tesla coil is"
STRONG = "Once upon a time there was a little girl who"


def load(path, device):
    for attempt in (1, 2):
        try:
            shutil.copy2(path, SCRATCH)
            checkpoint = torch.load(SCRATCH, map_location=device, mmap=True, weights_only=True)
            break
        except Exception as error:
            if attempt == 2:
                raise
            print(f"  checkpoint busy ({error}); retrying in 30s")
            time.sleep(30)
    model = DarkGLM(v2_config()).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    step = int(checkpoint.get("step", -1))
    del checkpoint
    return model, step


@torch.no_grad()
def run(model, tokenizer, prompt, device, tokens, penalty, temperature, seed=0):
    config = v2_config()
    ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    start = ids.shape[1]
    generator = torch.Generator().manual_seed(seed) if temperature > 0 else None

    for _ in range(tokens):
        logits = model(ids[:, -config.context_length:])[:, -1, :]

        if penalty and penalty != 1.0:
            # Generated tokens ONLY. Including the prompt is the documented anti-grounding bug.
            for token_id in set(ids[0, start:].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= penalty
                else:
                    logits[0, token_id] *= penalty

        if temperature <= 0:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            values, indices = torch.topk(logits / temperature, 50)
            probabilities = torch.softmax(values, dim=-1)
            nxt = indices.gather(-1, torch.multinomial(probabilities, num_samples=1,
                                                      generator=generator))
        ids = torch.cat([ids, nxt], dim=1)

    generated = [int(t) for t in ids[0, start:]]
    return generated, tokenizer.decode(generated)


def repetition(token_ids, n=6):
    if len(token_ids) <= n:
        return 0.0
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    counts = Counter(grams)
    return sum(c - 1 for c in counts.values() if c > 1) / len(grams)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/pretrain_v2_latest.pt")
    parser.add_argument("--tokens", type=int, default=60)
    args = parser.parse_args()

    device = "cpu"
    model, step = load(Path(args.checkpoint), device)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    print(f"checkpoint step {step:,}\n")

    print("=" * 78)
    print("REPETITION PENALTY on a weak-domain prompt (greedy, so nothing else varies)")
    print("=" * 78)
    print(f"prompt: {WEAK!r}\n")
    for penalty in (1.0, 1.1, 1.3, 1.5, 2.0):
        ids, text = run(model, tokenizer, WEAK, device, args.tokens, penalty, 0.0)
        label = "none" if penalty == 1.0 else f"{penalty}"
        marker = "  <- repo's serving default" if penalty == 1.3 else ""
        print(f"  penalty {label:<5} repetition {repetition(ids):.2f}{marker}")
        print(f"    {text.strip()[:150]}")
    print()

    print("=" * 78)
    print("TEMPERATURE instead (no penalty) - randomness, not repeat-awareness")
    print("=" * 78)
    for temperature in (0.3, 0.8, 1.2):
        ids, text = run(model, tokenizer, WEAK, device, args.tokens, 1.0, temperature)
        print(f"  temperature {temperature:<4} repetition {repetition(ids):.2f}")
        print(f"    {text.strip()[:150]}")
    print()

    print("=" * 78)
    print("THE COST - the same penalty on a prompt that was never looping")
    print("=" * 78)
    print(f"prompt: {STRONG!r}\n")
    for penalty in (1.0, 1.3, 2.0):
        ids, text = run(model, tokenizer, STRONG, device, args.tokens, penalty, 0.0)
        label = "none" if penalty == 1.0 else f"{penalty}"
        print(f"  penalty {label:<5} repetition {repetition(ids):.2f}")
        print(f"    {text.strip()[:150]}")

    print()
    print("=" * 78)
    print("Read the TEXT, not just the repetition number. A penalty removes the loop by")
    print("forcing a different token - it cannot supply knowledge the model does not have,")
    print("so the loop is replaced by fluent invention. Onboarding section 8 records exactly")
    print("this: FIRST_TOKEN_ECHO_PENALTY suppressed a correct-but-dull answer and produced")
    print("'photorespiration' and 'Robert Shakespeare' instead.")

    SCRATCH.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
