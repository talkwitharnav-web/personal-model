"""v2 pretraining: pre-norm + RoPE + residual scaling, on the mixed corpus, at the measured LR.

Everything here that differs from v1 was measured, not assumed:

  architecture   v2 (pre_norm + RoPE + scale_residual_init). Beat v1 by 0.342 validation on an
                 identical 3,000-step race, and at 9e-4 post-norm collapsed to the unigram floor
                 (7.4386 vs 7.4530) while v2 reached 3.8292.
  peak LR        1.0e-3. A five-point sweep on identical schedules gave 3e-4 -> 4.3947,
                 6e-4 -> 3.9317, 9e-4 -> 3.8292, 1.2e-3 -> 3.7771, 1.5e-3 -> 3.8872. The optimum
                 is bracketed at 1.2e-3; 1.0e-3 backs off deliberately because this run is ~43x
                 longer than the sweep, sits near peak proportionally longer, and therefore has
                 far more opportunity to hit an instability.
  warmup         2,000 steps (~1.5%). NOT the 500 from the sweep - patience and warmup must scale
                 with epoch length, never be copy-pasted (onboarding section 5).
  tokenizer      tokenizer_v2.json, refit on the real corpus, +9.42% compression.

Deliberate deviation from the smoke test: gradient clipping at 1.0. The sweep showed zero spikes
up to 1.5e-3 so it is not strictly needed, but a single bad step is unrecoverable over 30 hours
and clipping can only prevent a catastrophic update.

Resumable by design. A job that only writes at the end produces nothing if killed, and hours were
lost twice to exactly that (onboarding section 7).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.config import v2_config
from model.model import DarkGLM
from tools.memtrack import report

DATA_DIR = Path("data/pretrain")
CHECKPOINT_DIR = Path("checkpoints")
PHASE = "pretrain_v2"

BATCH_SIZE = 16
GRAD_ACCUM = 4
CONTEXT = 768
PEAK_LR = 1.0e-3
WARMUP = 2_000
MIN_LR_FRACTION = 0.1
GRAD_CLIP = 1.0
EVAL_EVERY = 2_000
EVAL_BATCHES = 20
CHECKPOINT_EVERY = 2_000
INIT_SEED = 1234
DATA_SEED = 99
VAL_SEED = 7

# Measured on THIS corpus with THIS tokenizer by tools/audit_pretrain_corpus.py: the
# cross-entropy of a model that knows only token frequencies and nothing about context.
# It is NOT the 7.4530 from the v1-vs-v2 smoke tests - that was the old tokenizer on a
# Wikipedia-only corpus. A floor copied from a different corpus makes this alarm meaningless.
UNIGRAM_FLOOR = 7.6689


def get_batch(data, generator, device, batch_size=BATCH_SIZE):
    top = len(data) - CONTEXT - 1
    starts = torch.randint(top, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i:i + CONTEXT].astype(np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + CONTEXT].astype(np.int64))
                     for i in starts])
    return x.to(device), y.to(device)


def lr_at(step, total_steps, peak_lr):
    if step < WARMUP:
        return peak_lr * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, total_steps - WARMUP)
    progress = min(progress, 1.0)
    return (peak_lr * MIN_LR_FRACTION
            + peak_lr * (1 - MIN_LR_FRACTION) * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1_600_000_000)
    parser.add_argument("--lr", type=float, default=PEAK_LR)
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any existing checkpoint and start over")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens_per_step = BATCH_SIZE * CONTEXT
    total_steps = args.tokens // tokens_per_step

    train_path = DATA_DIR / "pretrain_train.bin"
    val_path = DATA_DIR / "pretrain_validation.bin"
    for path in (train_path, val_path):
        if not path.exists():
            raise SystemExit(f"missing {path} - run data/tokenize_pretrain.py first")
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

    torch.manual_seed(INIT_SEED)
    config = v2_config()
    model = DarkGLM(config).to(device)
    model.gradient_checkpointing = True
    model.train()
    params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = CHECKPOINT_DIR / f"{PHASE}_latest.pt"
    best_path = CHECKPOINT_DIR / f"{PHASE}_best.pt"
    log_path = Path("tests/smoke_results") / f"{PHASE}_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # A fingerprint of everything that would invalidate a resume. Every script in this repo
    # auto-resumes from {phase}_latest.pt, and silently continuing a run whose data or LR changed
    # is a documented trap (onboarding section 5).
    fingerprint = {
        "tokens": args.tokens, "lr": args.lr, "warmup": WARMUP, "context": CONTEXT,
        "batch": BATCH_SIZE, "grad_accum": GRAD_ACCUM, "train_tokens": int(len(train_data)),
        "arch": "v2", "params": params,
    }

    start_step = 0
    best_val = float("inf")
    history = []
    if latest_path.exists() and not args.fresh:
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        saved = checkpoint.get("fingerprint", {})
        if saved != fingerprint:
            differences = {k: (saved.get(k), fingerprint[k])
                           for k in fingerprint if saved.get(k) != fingerprint[k]}
            raise SystemExit(
                f"{latest_path} was written by a DIFFERENT configuration: {differences}\n"
                f"Delete or rename it, or pass --fresh. Resuming would silently continue the "
                f"old run.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = checkpoint["step"] + 1
        best_val = checkpoint["best_val"]
        history = checkpoint.get("history", [])
        print(f"resumed from {latest_path} at step {start_step:,} (best val {best_val:.4f})")
        if start_step >= total_steps:
            raise SystemExit(f"start_step {start_step:,} already >= total {total_steps:,} - "
                             f"this run is finished. Delete the checkpoint to redo it.")

    train_rng = torch.Generator().manual_seed(DATA_SEED + start_step)

    @torch.no_grad()
    def validation_loss():
        model.eval()
        val_rng = torch.Generator().manual_seed(VAL_SEED)
        total = 0.0
        for _ in range(EVAL_BATCHES):
            x, y = get_batch(val_data, val_rng, device)
            with torch.autocast(device_type=device, dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            total += loss.item()
        model.train()
        return total / EVAL_BATCHES

    def save(path, step, value):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "best_val": best_val,
                    "history": history, "fingerprint": fingerprint, "config": vars(config)},
                   path)

    print(f"=== v2 PRETRAIN ===")
    print(f"device {device} | params {params:,} | vocab {config.vocab_size:,}")
    print(f"corpus {len(train_data):,} tokens | validation {len(val_data):,} tokens")
    print(f"{total_steps:,} steps x {tokens_per_step:,} tokens = {args.tokens:,} tokens")
    print(f"peak lr {args.lr:g} | warmup {WARMUP:,} | clip {GRAD_CLIP} | "
          f"eval/checkpoint every {EVAL_EVERY:,}")
    print(f"resume file {latest_path}")

    started = time.time()
    tokens_done = 0
    spikes = 0
    previous_val = None

    for step in range(start_step, total_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, total_steps, args.lr)

        x, y = get_batch(train_data, train_rng, device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        tokens_done += tokens_per_step

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = loss.item() * GRAD_ACCUM
        if not math.isfinite(train_loss):
            print(f"step {step} | NON-FINITE LOSS - stopping. Resume from {latest_path}.")
            save(latest_path, step, train_loss)
            raise SystemExit(1)

        if step % 200 == 0:
            elapsed = time.time() - started
            rate = tokens_done / max(elapsed, 1)
            remaining = (total_steps - step) * tokens_per_step / max(rate, 1) / 3600
            print(f"step {step:,}/{total_steps:,} | loss {train_loss:.4f} | "
                  f"lr {lr_at(step, total_steps, args.lr):.2e} | {rate:,.0f} tok/s | "
                  f"~{remaining:.1f}h left")

        if step % EVAL_EVERY == 0 or step == total_steps - 1:
            value = validation_loss()
            if previous_val is not None and value > previous_val + 0.5:
                spikes += 1
                print(f"  SPIKE {previous_val:.4f} -> {value:.4f}")
            previous_val = value
            history.append({"step": step, "val": round(value, 4),
                            "train": round(train_loss, 4)})
            marker = ""
            if value < best_val:
                best_val = value
                save(best_path, step, value)
                marker = " (best)"
            if value >= UNIGRAM_FLOOR - 0.15:
                marker += "  <-- AT UNIGRAM FLOOR, not using context"
            print(f"step {step:,} | validation {value:.4f}{marker}")
            log_path.write_text(json.dumps(
                {"history": history, "best_val": best_val, "spikes": spikes,
                 "fingerprint": fingerprint}, indent=2), encoding="utf-8")

        if step % CHECKPOINT_EVERY == 0 and step > start_step:
            save(latest_path, step, previous_val or float("inf"))

    save(latest_path, total_steps - 1, previous_val or float("inf"))
    hours = (time.time() - started) / 3600
    print(f"\ndone in {hours:.1f}h | best validation {best_val:.4f} | spikes {spikes}")
    print(f"best checkpoint {best_path}")
    report("(end of pretrain)")


if __name__ == "__main__":
    main()
