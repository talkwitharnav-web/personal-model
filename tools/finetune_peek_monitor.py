"""Peek at the model periodically DURING the fine-tune, so it cannot go astray unnoticed.

A 3.5-hour training run with nobody watching is how a collapse gets discovered at the end. The
fine-tune is the phase where that risk is highest: onboarding section 8 records SQuAD v2 falling
3.2648 -> 0.3726 on validation, textbook-looking, while the model had in fact collapsed into
refusing every question. Loss alone would not have caught it, and it did not.

So this samples the LIVE checkpoint every few minutes and writes what the model actually says,
plus the checks that catch the specific documented collapses:

  refusal      does it answer at all, or has it collapsed into silence / one stock reply
  echo         does "What is X?" come back as "X" (section 9's subject-echo shortcut)
  format       has it learned User:/Assistant: at all yet
  repetition   is it looping on text it should handle fluently

Reads a COPY of the checkpoint - the trainer rewrites it every 500 steps and a half-written read
would fail confusingly. CPU only, so it never competes with the run it is watching.
"""
import argparse
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model.config import v2_config
from model.model import DarkGLM
from tools.memtrack import system_free_mb

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Loading the model costs ~2 GB. The fine-tune it is watching needs ~1.7 GB, and this machine
# has 15.52 GB with Python dying SILENTLY when RAM runs short (onboarding section 2). Skipping a
# peek is free; killing the training run is not.
MIN_FREE_MB = 3200

TOKENIZER = Path("tokenizer/tokenizer_v2.json")
CHECKPOINT = Path("checkpoints/finetune_v2_latest.pt")
SCRATCH = Path("tests/finetune_peek.tmp")
LOG = Path("tests/finetune_peek_log.txt")
HISTORY = Path("tests/finetune_peek_history.json")

CONTEXT_TESLA = ("A Tesla coil is a resonant transformer circuit invented by Nikola Tesla in "
                 "1891. Penicillin is an antibiotic discovered by Alexander Fleming in 1928.")
CONTEXT_SUN = ("The Sun is the star at the centre of the Solar System. Its surface temperature "
               "is about 5,500 degrees Celsius, and it is by far the hottest object in the "
               "Solar System.")

# (label, question, context)
PROBES = [
    ("definition", "What is a Tesla coil?", CONTEXT_TESLA),
    ("binding", "Who discovered penicillin?", CONTEXT_TESLA),
    ("yesno", "Is the sun hot?", CONTEXT_SUN),
    ("maths", "Natalia sold clips to 48 friends in April, then half as many in May. "
              "How many clips did she sell altogether?", ""),
    ("chat", "What is your favourite colour?", ""),
    ("instruction", "Write one sentence about a dog.", ""),
]

STOP_SEQUENCES = ("\nUser:", "\nContext:", "\nAssistant:")


def note(message):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load(device, checkpoint_path=CHECKPOINT):
    for attempt in (1, 2, 3):
        try:
            shutil.copy2(checkpoint_path, SCRATCH)
            checkpoint = torch.load(SCRATCH, map_location=device, mmap=True, weights_only=True)
            model = DarkGLM(v2_config()).to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            step = int(checkpoint.get("step", -1))
            best = checkpoint.get("best_val", float("nan"))
            del checkpoint
            return model, step, best
        except Exception as error:
            if attempt == 3:
                raise
            time.sleep(20)


@torch.no_grad()
def answer(model, tokenizer, question, context, device, tokens=45):
    config = v2_config()
    prompt = (f"User: {question}\n\nContext: {context}\nAssistant:" if context
              else f"User: {question}\nAssistant:")
    ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    start = ids.shape[1]
    endoftext = tokenizer.token_to_id("<|endoftext|>")
    stopped = False
    for _ in range(tokens):
        logits = model(ids[:, -config.context_length:])[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        if nxt.item() == endoftext:
            stopped = True
            break
        ids = torch.cat([ids, nxt], dim=1)
        text = tokenizer.decode(ids[0, start:].tolist())
        if any(stop in text for stop in STOP_SEQUENCES):
            break
    text = tokenizer.decode(ids[0, start:].tolist())
    for stop in STOP_SEQUENCES:
        if stop in text:
            text = text.split(stop)[0]
    return text.strip(), stopped


def repetition(text, n=5):
    words = text.lower().split()
    if len(words) <= n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    return sum(c - 1 for c in counts.values() if c > 1) / len(grams)


def check(model, tokenizer, device):
    answers = {}
    for label, question, context in PROBES:
        text, stopped = answer(model, tokenizer, question, context, device)
        answers[label] = {"text": text, "stopped": stopped}

    blank = sum(1 for a in answers.values() if len(a["text"].strip()) < 2)
    stopped = sum(1 for a in answers.values() if a["stopped"])
    identical = len({a["text"] for a in answers.values()}) == 1
    echo = "tesla coil" in answers["definition"]["text"].lower()[:16]
    loops = max(repetition(a["text"]) for a in answers.values())

    warnings = []
    if blank >= len(PROBES) - 1:
        warnings.append("REFUSAL COLLAPSE: almost every answer is empty")
    if identical and len(PROBES) > 1:
        warnings.append("MODE COLLAPSE: every question gets the identical answer")
    if echo:
        warnings.append("SUBJECT ECHO: the definition answer starts by repeating the subject")
    if loops > 0.5:
        warnings.append(f"LOOPING: repeated 5-grams at {loops:.2f}")
    return answers, warnings, {"blank": blank, "stopped": stopped, "loops": round(loops, 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--every", type=int, default=900, help="seconds between peeks")
    parser.add_argument("--max-hours", dest="max_hours", type=float, default=6.0)
    parser.add_argument("--once", action="store_true")
    # So the load -> probe -> warn path can be exercised against an existing checkpoint before
    # the fine-tune exists. An untested monitor is worse than none: it fails at the first real
    # use, which is exactly when nobody is watching.
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    device = "cpu"
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    started = time.time()
    history = []

    note(f"fine-tune peek monitor started | every {args.every}s | max {args.max_hours}h "
         f"| watching {checkpoint_path}")

    while True:
        if not checkpoint_path.exists():
            note(f"waiting for {checkpoint_path} ...")
        else:
            free = system_free_mb()
            if free is not None and free < MIN_FREE_MB:
                note(f"SKIPPING this peek - only {free:,.0f} MB free (need {MIN_FREE_MB:,}). "
                     f"Never risk the run being watched.")
            else:
                try:
                    model, step, best = load(device, checkpoint_path)
                    answers, warnings, stats = check(model, tokenizer, device)
                    free_note = f" | free RAM {free:,.0f} MB" if free is not None else ""
                    note(f"step {step:,} | best val {best:.4f} | blank {stats['blank']}/"
                         f"{len(PROBES)} | stopped cleanly {stats['stopped']}/{len(PROBES)} | "
                         f"max loop {stats['loops']}{free_note}")
                    for label, data in answers.items():
                        note(f"    {label:<12} {data['text'][:110]!r}")
                    for warning in warnings:
                        note(f"  ** {warning}")
                    if not warnings:
                        note("    no collapse signals")
                    history.append({"step": step, "best_val": best, **stats,
                                    "warnings": warnings})
                    HISTORY.write_text(json.dumps(history, indent=2), encoding="utf-8")
                    del model
                except Exception as error:
                    note(f"peek failed ({type(error).__name__}: {error}) - will retry")

        if args.once:
            break
        if (time.time() - started) / 3600 > args.max_hours:
            note("max hours reached, stopping monitor")
            break
        time.sleep(args.every)

    SCRATCH.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
