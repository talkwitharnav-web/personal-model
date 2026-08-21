"""A wide peek: the familiar prompts plus new ones probing capabilities never tested.

Every peek so far has used the same six prompts, which measures continuity well and coverage
badly. Six prompts cannot distinguish "the model is weak" from "those six happen to be hard".

Grouped so the results say something specific rather than being a wall of text:

  continuity   the original six, so today's output is comparable to earlier peeks
  narrative    TinyStories register - 15% of the corpus, and where the model is strongest
  wikipedia    the article openings that dominate 80% of the corpus, by article TYPE
  everyday     facts a child knows, which do NOT appear as Wikipedia article openings and so
               test whether anything generalised beyond the corpus's phrasing
  structure    lists, dates, categories - the shape of Wikipedia rather than its content
  continuation mid-sentence and mid-paragraph, testing local coherence rather than recall

Sample AND greedy for each: a sample is one roll of the dice, greedy is what the model most
believes. Reading only one of them is how a peek misleads.
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
SCRATCH = Path("tests/widepeek_checkpoint.tmp")

GROUPS = {
    "continuity (the original six)": [
        "The Tesla coil is",
        "Photosynthesis is the process by which",
        "William Shakespeare was an English",
        "The capital city of France is",
        "Once upon a time there was a little girl who",
        "The Second World War began in",
    ],
    "narrative (TinyStories register)": [
        "Tom had a red ball. He",
        "The little cat was sad because",
        "Lily asked her mum, \"Can I",
        "One day a boy found a box in the garden. Inside the box",
    ],
    "wikipedia by article type": [
        "Paris is the capital and most populous city of",          # place
        "Albert Einstein was a German-born",                        # person, famous
        "The blue whale is a marine mammal that",                   # species
        "Python is a high-level programming language",              # technology
        "The Battle of Hastings was fought in",                     # event
        "Titanic is a 1997 American",                               # film
        "Mount Everest is the highest",                             # geography superlative
    ],
    "everyday knowledge (not Wikipedia phrasing)": [
        "The sky is",
        "Cats have four",
        "Water freezes at",
        "There are seven days in a",
        "The opposite of hot is",
        "Two plus two equals",
    ],
    "structure (the shape of Wikipedia)": [
        "Early life\nHe was born",
        "See also\n",
        "The film received",
        "According to the 2011 census, the population was",
    ],
    "continuation (local coherence, no recall)": [
        "She opened the door slowly, and inside she saw",
        "The company was founded in 1998 and quickly grew to",
        "Although the weather was cold, the children",
    ],
}


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
    best = checkpoint.get("best_val", float("nan"))
    del checkpoint
    return model, step, best


@torch.no_grad()
def generate(model, tokenizer, prompt, device, tokens, temperature, top_k=50, seed=None):
    config = v2_config()
    ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    start = ids.shape[1]
    endoftext = tokenizer.token_to_id("<|endoftext|>")
    generator = torch.Generator().manual_seed(seed) if (temperature > 0 and seed is not None) \
        else None
    for _ in range(tokens):
        logits = model(ids[:, -config.context_length:])[:, -1, :]
        if temperature <= 0:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            values, indices = torch.topk(logits / temperature, top_k)
            probabilities = torch.softmax(values, dim=-1)
            nxt = indices.gather(-1, torch.multinomial(probabilities, num_samples=1,
                                                      generator=generator))
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == endoftext:
            break
    return tokenizer.decode(ids[0, start:].tolist())


def wrap(text, width, indent):
    text = " ".join(text.split())
    lines = []
    while text:
        if len(text) <= width:
            lines.append(text)
            break
        cut = text.rfind(" ", 0, width)
        cut = cut if cut > 0 else width
        lines.append(text[:cut])
        text = text[cut:].lstrip()
    return ("\n" + " " * indent).join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/pretrain_v2_latest.pt")
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = "cpu"
    model, step, best = load(Path(args.checkpoint), device)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    total = 130_208
    print(f"checkpoint step {step:,}/{total:,} ({step/total*100:.1f}%) | "
          f"best validation {best:.4f}")
    print(f"{sum(len(v) for v in GROUPS.values())} prompts, sample (T={args.temperature}) "
          f"and greedy for each\n")

    for group, prompts in GROUPS.items():
        print("=" * 92)
        print(group.upper())
        print("=" * 92)
        for prompt in prompts:
            shown = prompt.replace("\n", "\\n")
            sampled = generate(model, tokenizer, prompt, device, args.tokens,
                               args.temperature, seed=0)
            greedy = generate(model, tokenizer, prompt, device, args.tokens, 0.0)
            print(f"\n  {shown!r}")
            print(f"    sample: {wrap(sampled, 78, 12)}")
            print(f"    greedy: {wrap(greedy, 78, 12)}")
        print()

    SCRATCH.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
