"""Is a peek regression real, or just sampling noise?

Peeks are read one sample at a time at temperature 0.8, and conclusions get drawn from them:
"Shakespeare went from playwright back to son of Prince Rupert, it got worse". That claim is
untested. One unlucky early token sends a whole continuation elsewhere, so a single sample cannot
distinguish a model that got worse from a model that rolled badly.

This runs the SAME prompt many times at one checkpoint and reports how often a wanted word
appears, plus the greedy (temperature 0) continuation - what the model most believes, with no
sampling at all. Greedy is the honest answer to "did it regress"; the sample rate says how noisy
the peek was.

CPU, and copies the checkpoint first, so it is safe while training holds the GPU.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model.config import v2_config
from model.model import DarkGLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOKENIZER = Path("tokenizer/tokenizer_v2.json")
SCRATCH = Path("tests/variance_checkpoint.tmp")

# (prompt, words that would mean the model got it right)
CASES = [
    ("William Shakespeare was an English", ["playwright", "poet", "dramatist", "writer"]),
    ("The Second World War began in", ["1939"]),
    ("The Tesla coil is", ["transformer", "coil", "electr", "resonan"]),
    ("The capital city of France is", ["paris"]),
]


def load(checkpoint_path, device):
    for attempt in (1, 2):
        try:
            shutil.copy2(checkpoint_path, SCRATCH)
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
    best = checkpoint.get("best_val", float("nan"))
    del checkpoint
    return model, step, best


@torch.no_grad()
def continue_text(model, tokenizer, prompt, device, tokens, temperature, top_k, generator=None):
    config = v2_config()
    ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    start = ids.shape[1]
    endoftext = tokenizer.token_to_id("<|endoftext|>")
    for _ in range(tokens):
        logits = model(ids[:, -config.context_length:])[:, -1, :]
        if temperature <= 0:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            values, indices = torch.topk(logits, top_k)
            probabilities = torch.softmax(values, dim=-1)
            nxt = indices.gather(-1, torch.multinomial(probabilities, num_samples=1,
                                                      generator=generator))
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == endoftext:
            break
    return tokenizer.decode(ids[0, start:].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/pretrain_v2_latest.pt")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", dest="top_k", type=int, default=50)
    args = parser.parse_args()

    device = "cpu"
    model, step, best = load(Path(args.checkpoint), device)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    print(f"checkpoint step {step:,} | best validation {best:.4f}")
    print(f"{args.samples} samples per prompt at temperature {args.temperature}, "
          f"plus greedy\n")

    for prompt, wanted in CASES:
        generator = torch.Generator().manual_seed(0)
        hits = 0
        firsts = []
        for _ in range(args.samples):
            text = continue_text(model, tokenizer, prompt, device, args.tokens,
                                 args.temperature, args.top_k, generator)
            lowered = text.lower()
            if any(word in lowered for word in wanted):
                hits += 1
            firsts.append(" ".join(text.split()[:6]))

        greedy = continue_text(model, tokenizer, prompt, device, args.tokens, 0.0, args.top_k)
        greedy_ok = any(word in greedy.lower() for word in wanted)

        print(f"--- {prompt!r}   (want any of {wanted})")
        print(f"    sampled : {hits}/{args.samples} contained a wanted word")
        print(f"    greedy  : {'HIT ' if greedy_ok else 'miss'} {greedy.strip()[:110]!r}")
        for sample in firsts[:4]:
            print(f"      e.g. {sample!r}")
        print()

    SCRATCH.unlink(missing_ok=True)
    print("Greedy is what the model most believes. A low sampled rate with a greedy hit means")
    print("the peek was noisy, not that the model regressed.")


if __name__ == "__main__":
    main()
