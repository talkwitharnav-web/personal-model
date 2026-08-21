"""Score the model on REAL held-out yes/no examples, not hand-written ones.

`tools/yesno_knowledge_probe.py` writes its own short, tidy contexts. The training data does not
look like that at all - it is two to five Wikipedia passages joined together, one of which happens
to bear on the question. A model can be perfectly competent on the real distribution and still look
broken on a probe written in a different register, and concluding "the model cannot do yes/no" from
the probe alone would be exactly the mistake of measuring the wrong population.

So this reads `data/yesno/validation.txt` - held out, never trained on, same generator as the
training set - and reports accuracy. It also splits by whether the example carries a Context block,
because the 25% no-context share is the part that has to teach unaided recall.
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


def load(checkpoint_path, device):
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer_v2.json")
    model = DarkGLM(v2_config())
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    return model, tokenizer, blob.get("step", -1)


def read_examples(path, limit):
    rows = []
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for block in text.split("<|endoftext|>"):
        block = block.strip()
        if not block:
            continue
        match = re.search(r"\nAssistant:", block)
        if not match:
            continue
        prompt, answer = block[:match.start()], block[match.end():].strip()
        has_context = "\nContext:" in prompt or "\n\nContext:" in prompt
        rows.append((prompt + "\nAssistant:", answer, has_context))
        if len(rows) >= limit:
            break
    return rows


@torch.no_grad()
def predict(model, tokenizer, prompt, device):
    ids = tokenizer.encode(prompt).ids[-760:]
    logits = model(torch.tensor([ids], device=device))[0, -1]
    probs = F.softmax(logits.float(), dim=-1)

    def mass(word):
        best = 0.0
        for variant in (f" {word}", word):
            encoded = tokenizer.encode(variant).ids
            if encoded:
                best = max(best, probs[encoded[0]].item())
        return best

    yes, no = mass("Yes"), mass("No")
    return ("Yes" if yes >= no else "No"), yes / (yes + no) if (yes + no) > 0 else 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/finetune_v2_best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()

    model, tokenizer, step = load(args.checkpoint, args.device)
    rows = read_examples("data/yesno/validation.txt", args.limit)
    print(f"checkpoint {args.checkpoint} (step {step:,})")
    print(f"{len(rows)} held-out examples from data/yesno/validation.txt\n")

    buckets = {True: [0, 0], False: [0, 0]}      # has_context -> [correct, total]
    said_yes = 0
    for prompt, answer, has_context in rows:
        guess, _ = predict(model, tokenizer, prompt, args.device)
        said_yes += guess == "Yes"
        buckets[has_context][1] += 1
        buckets[has_context][0] += guess == answer

    total_correct = sum(b[0] for b in buckets.values())
    total = sum(b[1] for b in buckets.values())
    print(f"  WITH context:  {buckets[True][0]:>4}/{buckets[True][1]:<4} "
          f"= {100*buckets[True][0]/max(buckets[True][1],1):5.1f}%")
    print(f"  NO context:    {buckets[False][0]:>4}/{buckets[False][1]:<4} "
          f"= {100*buckets[False][0]/max(buckets[False][1],1):5.1f}%")
    print(f"  overall:       {total_correct:>4}/{total:<4} = {100*total_correct/max(total,1):5.1f}%"
          f"   (chance 50.0%)")
    print(f"  answered Yes {100*said_yes/max(total,1):.1f}% of the time "
          f"(a healthy model sits near 50% on a 50/50 set)")

    print()
    accuracy = total_correct / max(total, 1)
    if accuracy > 0.70:
        print("  -> The model CAN do this task on its own distribution. A flat result on the")
        print("     hand-written probe is a register mismatch in the probe, not a broken model.")
    elif accuracy > 0.58:
        print("  -> Above chance but weak. Real ability, still forming - re-measure later.")
    else:
        print("  -> At chance on held-out data from its OWN generator. This is not a probe")
        print("     artifact; the capability is genuinely absent so far.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
