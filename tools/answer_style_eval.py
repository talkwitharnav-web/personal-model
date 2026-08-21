"""Does the model ANSWER with context, or just extract a short span?

The RAG peek showed correct one-word answers for lookups ("Paris", "1939") but the same clipped
style everywhere else - "What is photosynthesis?" answered with "photosynthesis", "Do fish have
lungs?" answered with "lungfish", "How many legs does a cat have?" answered with "six" from a
passage that says four. That looks like SQuAD-style span extraction being applied to every question
shape regardless of what the question needs.

But a hand-written probe already produced a false alarm once today (see the yes/no register
mismatch), so the hypothesis gets tested on REAL held-out rows with their REAL contexts before
anyone acts on it. Reference answers come from the same generator, so length and echo can be
compared against what the data actually asks for.

Reported per dataset:
  len ratio    generated answer length / reference answer length. Near 1.0 is right; far below 1.0
               means the model is clipping.
  subject echo how often the answer merely repeats the question's subject - the section 9 defect
               that `prepare_definitions.py` was rewritten to remove.
  first-token  whether it opens the way the references open.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.model import DarkGLM
from training.finetune_v2 import v2_config

STOP = set("what is a an the of give me explain define can you tell about describe do does "
           "in simple terms could please".split())


def load(checkpoint_path, device):
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer_v2.json")
    model = DarkGLM(v2_config())
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    return model, tokenizer, blob.get("step", -1), blob.get("best_val", float("nan"))


def read(path, limit):
    rows = []
    for block in Path(path).read_text(encoding="utf-8", errors="replace").split("<|endoftext|>"):
        block = block.strip()
        if not block:
            continue
        match = re.search(r"\nAssistant:", block)
        if not match:
            continue
        rows.append((block[:match.start()] + "\nAssistant:", block[match.end():].strip()))
        if len(rows) >= limit:
            break
    return rows


@torch.no_grad()
def generate(model, tokenizer, prompt, device, tokens=40):
    endoftext = tokenizer.token_to_id("<|endoftext|>")
    ids = tokenizer.encode(prompt).ids[-720:]
    out = []
    for _ in range(tokens):
        logits = model(torch.tensor([ids], device=device))[0, -1].float()
        for token in set(out):
            logits[token] /= 1.3
        nxt = int(torch.argmax(logits))
        if nxt == endoftext:
            break
        out.append(nxt)
        ids.append(nxt)
    return tokenizer.decode(out).strip()


def subject_of(prompt):
    question = prompt.split("\n")[0].replace("User:", "").strip().rstrip("?.")
    words = [w for w in re.findall(r"[A-Za-z0-9']+", question) if w.lower() not in STOP]
    return " ".join(words).lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/finetune_v2_best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--datasets", default="definitions,squad,sciq")
    args = parser.parse_args()

    model, tokenizer, step, best = load(args.checkpoint, args.device)
    print(f"checkpoint {args.checkpoint} (step {step:,}, validation {best:.4f})\n")

    for name in args.datasets.split(","):
        path = Path("data") / name / "validation.txt"
        if not path.exists():
            print(f"{name}: no validation.txt, skipped")
            continue
        rows = read(path, args.limit)
        ratios, echoes, gen_lens, ref_lens = [], 0, [], []
        samples = []
        for prompt, reference in rows:
            answer = generate(model, tokenizer, prompt, args.device)
            g, r = len(answer.split()), len(reference.split())
            gen_lens.append(g)
            ref_lens.append(r)
            ratios.append(g / max(r, 1))
            subject = subject_of(prompt)
            if subject and answer.lower().strip(" .") in (subject, subject.rstrip("s")):
                echoes += 1
            if len(samples) < 3:
                samples.append((prompt.split("\n")[0][:70], reference[:70], answer[:70]))

        n = max(len(rows), 1)
        print(f"=== {name}  ({len(rows)} held-out rows) ===")
        print(f"    generated length {sum(gen_lens)/n:5.1f} words | "
              f"reference {sum(ref_lens)/n:5.1f} words | ratio {sum(ratios)/n:4.2f}")
        print(f"    answer is just the question's subject: {echoes}/{len(rows)} "
              f"({100*echoes/n:.0f}%)")
        for question, reference, answer in samples:
            print(f"      Q    {question}")
            print(f"      ref  {reference}")
            print(f"      got  {answer}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
