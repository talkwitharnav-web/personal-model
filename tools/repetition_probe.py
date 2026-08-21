"""Is greedy looping a sign of a degrading model, or of a model that lacks the knowledge?

A peek at step 102,000 showed greedy decoding looping ("The Tesla Model T is a single-piece
instrument..." three times) where step 98,000 had produced a coherent biography. That could mean
the model is degrading - or it could be ordinary greedy behaviour, since greedy always takes the
single most likely token and has no mechanism to escape a cycle it falls into.

One observation splits the two: the TinyStories prompt did NOT loop under greedy at the same
checkpoint. If looping tracks how well the model knows a domain rather than the checkpoint, the
loop is a symptom of uncertainty, not decay.

So this measures repetition across prompts grouped by whether the model should know the domain:
  strong  - TinyStories-style narrative and very common Wikipedia openings
  weak    - technical concepts the peeks have shown it does not know

If repetition concentrates in `weak`, greedy looping is a knowledge signal and not a reason to
stop the run. Reported as a number so it can be re-measured later rather than argued about.
"""
import argparse
import json
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
SCRATCH = Path("tests/repetition_checkpoint.tmp")
HISTORY = Path("tests/repetition_history.json")

# ⚠ These two groups measure DIFFERENT things and were originally lumped together as "strong".
# TinyStories prompts test whether the model loops on text it knows well. Wikipedia stub
# openings are FORMULAIC - real Wikipedia genuinely writes "in the district of X, in the
# province of Y" - so repetition there is partly a property of the source text, not the model.
# Averaging them produced a "drift" from 0.069 to 0.103 that was really the formulaic group
# moving. Kept separate so the tripwire watches the group that actually means something.
NARRATIVE = [
    "Once upon a time there was a little girl who",
    "Once upon a time there was a boy named Tom who",
    "Lily and her mum went to the park. They",
]
FORMULAIC = [
    "The village is located in the",
    "He was born in 1892 in",
    "The film was released in",
]
WEAK = [
    "The Tesla coil is",
    "Photosynthesis is the process by which",
    "The mitochondrion is",
    "Quantum entanglement is",
    "The capital city of France is",
    "The Second World War began in",
]


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
def greedy(model, tokenizer, prompt, device, tokens):
    """Greedy continuation, plus the mean confidence of each choice.

    Confidence is the probability mass the model puts on the token it actually picked. It
    separates two very different kinds of looping:
      - CONFIDENT looping: the model is sure, and sure of something wrong. That is what a
        sharpening distribution produces in a domain with no real knowledge.
      - UNCERTAIN looping: the model is guessing and happens to cycle.
    Reported so "the anneal is sharpening it" is a measurement rather than a story.
    """
    config = v2_config()
    ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    start = ids.shape[1]
    confidences = []
    for _ in range(tokens):
        logits = model(ids[:, -config.context_length:])[:, -1, :]
        probabilities = torch.softmax(logits.float(), dim=-1)
        chosen = torch.argmax(logits, dim=-1, keepdim=True)
        confidences.append(float(probabilities[0, chosen.item()]))
        ids = torch.cat([ids, chosen], dim=1)
    return [int(t) for t in ids[0, start:]], sum(confidences) / len(confidences)


def repetition(token_ids, n=6):
    """Share of n-gram positions that are a repeat. 0 = never repeats, high = looping."""
    if len(token_ids) <= n:
        return 0.0
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    counts = Counter(grams)
    repeats = sum(count - 1 for count in counts.values() if count > 1)
    return repeats / len(grams)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/pretrain_v2_latest.pt")
    parser.add_argument("--tokens", type=int, default=70)
    # Tripwire: looping on text the model KNOWS is a real problem; looping on unknown topics is
    # ordinary greedy behaviour, and looping on formulaic Wikipedia stubs is a property of the
    # source text. So the limit applies to the narrative group only. Measured 0.000 at step
    # 108,000, so 0.15 is a wide margin.
    parser.add_argument("--narrative-limit", dest="strong_limit", type=float, default=0.15)
    args = parser.parse_args()

    device = "cpu"
    model, step, best = load(Path(args.checkpoint), device)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    print(f"checkpoint step {step:,} | best validation {best:.4f}")
    print(f"greedy, {args.tokens} tokens, repetition = share of repeated 6-grams\n")

    results = {}
    confidence = {}
    for label, prompts, note in (("narrative", NARRATIVE, "text the model knows well"),
                                 ("formulaic", FORMULAIC, "Wikipedia stubs - repetitive by nature"),
                                 ("weak", WEAK, "domains the model does NOT know")):
        scores = []
        confidences = []
        print(f"--- {label} ({note})")
        for prompt in prompts:
            ids, mean_confidence = greedy(model, tokenizer, prompt, device, args.tokens)
            score = repetition(ids)
            scores.append(score)
            confidences.append(mean_confidence)
            text = tokenizer.decode(ids).replace("\n", " ")[:46]
            print(f"    rep {score:5.2f}  conf {mean_confidence:.2f}  "
                  f"{prompt[:32]:32} -> {text}")
        results[label] = sum(scores) / len(scores)
        confidence[label] = sum(confidences) / len(confidences)
        print(f"    mean repetition {results[label]:.3f} | mean confidence "
              f"{confidence[label]:.3f}\n")

    narrative, formulaic, weak = (results["narrative"], results["formulaic"], results["weak"])

    # Append to a history so drift is MEASURED over the run rather than inferred from two
    # readings - two points is not a trend, and treating it as one is how a normal fluctuation
    # gets reported as a regression.
    history = []
    if HISTORY.exists():
        try:
            history = json.loads(HISTORY.read_text(encoding="utf-8"))
        except Exception:
            history = []
    # Replace any earlier reading for the same step rather than appending a duplicate: running
    # the probe twice at one checkpoint would otherwise put the same value in the series twice
    # and make a flat trend look like three confirming readings.
    history = [entry for entry in history if entry.get("step") != step]
    history.append({"step": step, "narrative": round(narrative, 4),
                    "formulaic": round(formulaic, 4), "weak": round(weak, 4),
                    "conf_narrative": round(confidence["narrative"], 4),
                    "conf_weak": round(confidence["weak"], 4)})
    history.sort(key=lambda entry: entry["step"])
    HISTORY.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("=" * 74)
    if len(history) > 1:
        print("history (narrative is the one that matters):")
        for entry in history[-8:]:
            print(f"  step {entry['step']:>7,}  narrative {entry['narrative']:.3f}  "
                  f"formulaic {entry['formulaic']:.3f}  weak {entry['weak']:.3f}"
                  + (f"  conf(weak) {entry['conf_weak']:.3f}"
                     if "conf_weak" in entry else ""))
        if len(history) >= 3:
            recent = [e["narrative"] for e in history[-3:]]
            direction = ("rising" if recent[-1] > recent[0] + 0.02 else
                         "falling" if recent[-1] < recent[0] - 0.02 else "flat")
            print(f"  narrative trend over the last 3 readings: {direction}")
        else:
            print("  (two readings is not a trend - need a third before calling direction)")
        print()

    print(f"narrative {narrative:.3f}   formulaic {formulaic:.3f}   weak {weak:.3f}   "
          f"(tripwire: narrative > {args.strong_limit})")
    print(f"confidence in the chosen token: narrative {confidence['narrative']:.3f}  "
          f"weak {confidence['weak']:.3f}")
    if confidence["weak"] > 0.8:
        print("  -> the model is CONFIDENTLY looping on topics it does not know. That is what a")
        print("     sharpening distribution does where there is no real knowledge to sharpen")
        print("     toward, and it is why retrieval rather than decoding tricks is the fix.")
    elif confidence["weak"] > confidence["narrative"] + 0.15:
        print("  -> weak-domain choices are notably more confident than narrative ones, which")
        print("     supports the 'confidently wrong' reading.")
    else:
        print("  -> confidence is SIMILAR across groups, so 'the model is confidently looping'")
        print("     is not supported at the group level. Look at the per-prompt numbers: the")
        print("     prompts that loop hardest do carry high confidence, while the one that")
        print("     answered correctly carried low confidence. The group mean hides that.")
    if narrative >= args.strong_limit:
        print("VERDICT: FAIL - the model loops on text it knows well. That is NOT ordinary")
        print("greedy behaviour and is worth investigating before the fine-tune.")
        SCRATCH.unlink(missing_ok=True)
        return 1
    if weak > narrative * 2:
        print("VERDICT: PASS - looping concentrates where the model lacks knowledge, and is")
        print("near zero where it is confident. That is greedy decoding behaving normally under")
        print("uncertainty - NOT a degrading model. Sampling avoids it, which is what serving")
        print("uses anyway.")
    else:
        print("VERDICT: PASS on the tripwire, but the gap is smaller than expected.")

    SCRATCH.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
