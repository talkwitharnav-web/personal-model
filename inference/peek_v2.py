"""Peek at the v2 model mid-pretraining, without disturbing the run.

inference/peek.py cannot load this checkpoint: it builds a v1 DarkGLMConfig, reads the v1
tokenizer, and expects a bare state_dict. This one is v2 architecture, tokenizer_v2, and a
training checkpoint dict.

⚠ EXPECT TEXT CONTINUATION, NOT ANSWERS. This model has only been pretrained. It has never seen
"User: ... Assistant:" - that format arrives in the fine-tune (step 8). Prompting it like a chat
model would make it look far worse than it is, so the prompts here are plain prose openings and
the right question is "does it continue coherently and stay on topic".

Two safety measures, because a 30-hour run owns the machine:
  - CPU only. Training is holding ~7.7 GB of the 8.19 GB of VRAM.
  - The checkpoint is COPIED first, then loaded with mmap. The trainer rewrites the 911 MB file
    every 2,000 steps, so reading it in place risks catching a half-written file, and a plain
    torch.load would pull the optimizer state (two extra moment tensors per parameter) into a
    machine with under 2 GB free.
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
from tools.memtrack import report

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKPOINT = Path("checkpoints/pretrain_v2_latest.pt")
TOKENIZER = Path("tokenizer/tokenizer_v2.json")
SCRATCH = Path("tests/peek_checkpoint.tmp")

PROMPTS = [
    "The Tesla coil is",
    "Photosynthesis is the process by which",
    "William Shakespeare was an English",
    "The capital city of France is",
    "Once upon a time there was a little girl who",
    "The Second World War began in",
]


def load_model(device):
    # Copy first: the trainer rewrites this file every 2,000 steps and a half-written read would
    # fail confusingly. Retry once, since the copy itself can land mid-write.
    for attempt in (1, 2):
        try:
            shutil.copy2(CHECKPOINT, SCRATCH)
            checkpoint = torch.load(SCRATCH, map_location=device, mmap=True, weights_only=True)
            break
        except Exception as error:
            if attempt == 2:
                raise
            print(f"  checkpoint read failed ({error}); it is probably mid-write, retrying in 30s")
            time.sleep(30)

    step = int(checkpoint.get("step", -1))
    best = checkpoint.get("best_val", float("nan"))
    weights = checkpoint["model"]

    model = DarkGLM(v2_config()).to(device)
    model.load_state_dict(weights)
    model.eval()
    del checkpoint, weights
    return model, step, best


@torch.no_grad()
def generate(model, tokenizer, prompt, config, max_new_tokens, temperature, top_k, device):
    ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    endoftext = tokenizer.token_to_id("<|endoftext|>")
    for _ in range(max_new_tokens):
        logits = model(ids[:, -config.context_length:])[:, -1, :]
        if temperature <= 0:
            # Greedy: what the model MOST believes, with no sampling luck involved.
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            values, indices = torch.topk(logits, top_k)
            probabilities = torch.softmax(values, dim=-1)
            nxt = indices.gather(-1, torch.multinomial(probabilities, num_samples=1))
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == endoftext:
            break
    return tokenizer.decode(ids[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", dest="top_k", type=int, default=50)
    parser.add_argument("--prompt", action="append", default=None)
    args = parser.parse_args()

    device = "cpu"          # training owns the GPU
    if not CHECKPOINT.exists():
        raise SystemExit(f"missing {CHECKPOINT}")

    print(f"loading {CHECKPOINT} on {device} ...")
    model, step, best = load_model(device)
    total = 130_208
    print(f"checkpoint from step {step:,}/{total:,} ({step/total*100:.1f}% of pretraining) | "
          f"best validation {best:.4f}")
    report("(model loaded)")

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    config = v2_config()

    print("\nNOTE: pretrained only - this continues text, it cannot answer questions yet.")
    print("Each prompt shows one SAMPLE and the GREEDY continuation. Read the greedy one to")
    print("judge whether the model knows something: a sample is a 1-in-8 lottery, and reading")
    print("one as evidence of regression is a mistake this script used to invite.\n")
    for prompt in (args.prompt or PROMPTS):
        sampled = generate(model, tokenizer, prompt, config, args.tokens,
                           args.temperature, args.top_k, device)
        greedy = generate(model, tokenizer, prompt, config, args.tokens, 0.0,
                          args.top_k, device)
        print(f"--- {prompt!r}")
        print(f"  sample: {sampled.strip()}")
        print(f"  greedy: {greedy.strip()}\n")

    SCRATCH.unlink(missing_ok=True)
    report("(done)")


if __name__ == "__main__":
    main()
