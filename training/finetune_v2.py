"""v2 fine-tune: packed instruction data on top of the v2 pretrained model.

Differs from training/train_combined.py in ways that all follow from measurements:

  data       data/finetune_packed (154,059 rows) instead of data/combined (571,319 padded rows).
             Every position is real, so an epoch is 3.64x fewer steps for the same content.
  init       starts from checkpoints/pretrain_v2_best.pt, a training checkpoint dict, not a bare
             state_dict.
  arch       v2 (pre-norm + RoPE + residual scaling).
  lr         peak 1e-4 -> 1e-5 cosine. v1 used 3e-5 because post-norm could not take more; the
             sweep showed v2 stays stable far higher (it survives 9e-4 where post-norm collapses
             to the unigram floor). Still an order of magnitude below the 1e-3 pretrain peak,
             because fine-tuning on 118M tokens can wash out pretrained knowledge.
  patience   scaled to THIS run's epoch length. Copying a patience value between phases is a
             recorded bug: Wikipedia early-stopped at step 2,750 of 89,518 because patience=5 was
             carried over from a much shorter run (onboarding section 5).

The masking rules that must not be broken (section 5):
  - y comes from LABELS, not ids: x = ids[:, :-1], y = labels[:, 1:]
  - ignore_index=-100 must be passed to cross_entropy or the -100s are treated as real classes
  - row-based batching, never random-offset slicing, or offsets cut across the prompt/answer
    boundary
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

DATA_DIR = Path("data/finetune_packed")
CHECKPOINT_DIR = Path("checkpoints")
PRETRAINED = CHECKPOINT_DIR / "pretrain_v2_best.pt"
PHASE = "finetune_v2"

# 8 x 8 = the same 64 rows per optimiser step as 16 x 4, but half the peak VRAM.
# At 16 the run reserved 7.38 GB of the 7.95 GB this card has free and died in backward
# around step 150 - the logits tensor alone is batch x 767 x 32000, and cross-entropy
# upcasts it to fp32. Windows does not support expandable_segments, so there is no
# allocator-level fix available; the batch has to come down.
BATCH_SIZE = 8
GRAD_ACCUM = 8
PEAK_LR = 1.0e-4
MIN_LR_FRACTION = 0.1
WARMUP_FRACTION = 0.02          # 2% of the run, not a copied constant
GRAD_CLIP = 1.0
EVAL_EVERY = 500
EVAL_BATCHES = 30
CHECKPOINT_EVERY = 1000
PATIENCE_EVALS = 12             # 12 x 500 = 6,000 steps without improvement
INIT_SEED = 1234
DATA_SEED = 99
VAL_SEED = 7


def load_split(split_name):
    ids = np.load(DATA_DIR / f"{split_name}_ids.npy", mmap_mode="r")
    labels = np.load(DATA_DIR / f"{split_name}_labels.npy", mmap_mode="r")
    if ids.shape != labels.shape:
        raise SystemExit(f"{split_name}: ids {ids.shape} != labels {labels.shape}")
    return ids, labels


def get_batch(ids_source, labels_source, generator, device):
    rows = torch.randint(len(ids_source), (BATCH_SIZE,), generator=generator).numpy()
    rows.sort()                       # sorted reads are far kinder to a memory-mapped file
    ids_batch = ids_source[rows].astype(np.int64)
    labels_batch = labels_source[rows].astype(np.int64)
    # y from LABELS, not ids - this is the whole point of loss masking.
    x = torch.from_numpy(ids_batch[:, :-1]).to(device)
    y = torch.from_numpy(labels_batch[:, 1:]).to(device)
    return x, y


def lr_at(step, total_steps, peak_lr, warmup):
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = min((step - warmup) / max(1, total_steps - warmup), 1.0)
    return (peak_lr * MIN_LR_FRACTION
            + peak_lr * (1 - MIN_LR_FRACTION) * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=PEAK_LR)
    parser.add_argument("--fresh", action="store_true")
    # Smoke mode: a few steps into a scratch directory, to prove the loop, the backward pass,
    # the eval and the checkpoint save all work BEFORE committing hours to them. The fine-tune
    # has never been run, and "the data path is verified" is not the same as "a gradient has
    # flowed through it".
    parser.add_argument("--smoke", type=int, default=0,
                        help="run N steps into checkpoints/smoke/ and exit")
    args = parser.parse_args()

    smoke = args.smoke > 0
    checkpoint_dir = Path("checkpoints/smoke") if smoke else CHECKPOINT_DIR
    phase = f"{PHASE}_smoke" if smoke else PHASE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ids, train_labels = load_split("train")
    val_ids, val_labels = load_split("validation")

    steps_per_epoch = len(train_ids) // BATCH_SIZE
    total_steps = args.smoke if smoke else int(args.epochs * steps_per_epoch)
    warmup = max(2, int(total_steps * 0.1)) if smoke else max(50, int(total_steps
                                                                     * WARMUP_FRACTION))

    torch.manual_seed(INIT_SEED)
    config = v2_config()
    model = DarkGLM(config).to(device)
    model.gradient_checkpointing = True
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device)

    latest_path = checkpoint_dir / f"{phase}_latest.pt"
    best_path = checkpoint_dir / f"{phase}_best.pt"
    log_path = Path("tests/smoke_results") / f"{phase}_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = {
        "epochs": args.epochs, "lr": args.lr, "warmup": warmup, "batch": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM, "rows": int(len(train_ids)), "arch": "v2",
    }

    start_step = 0
    best_val = float("inf")
    stale = 0
    history = []

    if latest_path.exists() and not args.fresh:
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        saved = checkpoint.get("fingerprint", {})
        if saved != fingerprint:
            differences = {k: (saved.get(k), fingerprint[k])
                           for k in fingerprint if saved.get(k) != fingerprint[k]}
            raise SystemExit(f"{latest_path} was written by a DIFFERENT configuration: "
                             f"{differences}\nDelete it or pass --fresh.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = checkpoint["step"] + 1
        best_val = checkpoint["best_val"]
        stale = checkpoint.get("stale", 0)
        history = checkpoint.get("history", [])
        print(f"resumed from {latest_path} at step {start_step:,}")
        # Same trap as the fresh-start branch below, and it was missed here at first: the
        # checkpoint was loaded with map_location=device, so this dict holds a SECOND copy of
        # the model, optimizer and scaler state on the GPU. Without the del it stays live for
        # the entire run - measured as 4.89 G reserved on resume versus 3.97 G on a fresh
        # start, a 0.92 G gap that is exactly the 911 MB checkpoint.
        del checkpoint
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        if not PRETRAINED.exists():
            raise SystemExit(f"missing {PRETRAINED} - the pretrain must finish first")
        pretrained = torch.load(PRETRAINED, map_location=device, weights_only=True)
        model.load_state_dict(pretrained["model"])
        print(f"initialised from {PRETRAINED} "
              f"(pretrain step {pretrained.get('step', '?'):,}, "
              f"validation {pretrained.get('best_val', float('nan')):.4f})")
        del pretrained
        # The checkpoint was loaded onto the GPU, so its blocks stay in torch's caching
        # allocator after the del. The pretrain never carried that extra pool and it fits
        # in 8 GB only just, so hand the memory back before training starts allocating.
        if device == "cuda":
            torch.cuda.empty_cache()

    train_rng = torch.Generator().manual_seed(DATA_SEED + start_step)

    # Reset the high-water mark AFTER loading. `max_memory_reserved()` is a peak since process
    # start, so on a resume it reports the transient spike from holding the checkpoint and the
    # model on the GPU at once - 4.58 G against 3.97 G for an identical fresh start. That is a
    # measurement artifact, not a leak (the checkpoint is deleted and the cache emptied above),
    # but it makes the two runs look different when they are not. Reset so the number printed
    # during training describes training.
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def validation_loss():
        model.eval()
        val_rng = torch.Generator().manual_seed(VAL_SEED)
        total = 0.0
        for _ in range(EVAL_BATCHES):
            x, y = get_batch(val_ids, val_labels, val_rng, device)
            with torch.autocast(device_type=device, dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, config.vocab_size),
                                       y.reshape(-1), ignore_index=-100)
            total += loss.item()
        model.train()
        return total / EVAL_BATCHES

    def save(path, step):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "best_val": best_val,
                    "stale": stale, "history": history, "fingerprint": fingerprint}, path)

    EVAL = max(2, args.smoke // 2) if smoke else EVAL_EVERY
    CHECKPOINT_AT = max(2, args.smoke // 2) if smoke else CHECKPOINT_EVERY

    print(f"=== v2 FINE-TUNE{' (SMOKE)' if smoke else ''} ===")
    print(f"device {device} | train {len(train_ids):,} rows | validation {len(val_ids):,} rows")
    print(f"{args.epochs} epochs x {steps_per_epoch:,} = {total_steps:,} steps "
          f"| peak lr {args.lr:g} | warmup {warmup:,}")
    print(f"patience {PATIENCE_EVALS} evals ({PATIENCE_EVALS * EVAL:,} steps) "
          f"of a {steps_per_epoch:,}-step epoch")
    print(f"checkpoints -> {checkpoint_dir}")

    started = time.time()
    tokens_done = 0

    for step in range(start_step, total_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, total_steps, args.lr, warmup)

        x, y = get_batch(train_ids, train_labels, train_rng, device)
        valid = int((y != -100).sum())
        if valid == 0:
            # Every target masked would give a NaN loss. Verified absent from the packed data,
            # but a batch that cannot teach anything is not worth a step either way.
            continue

        with torch.autocast(device_type=device, dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size),
                                   y.reshape(-1), ignore_index=-100)
            loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        tokens_done += BATCH_SIZE * (config.context_length - 1)

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = loss.item() * GRAD_ACCUM
        if not math.isfinite(train_loss):
            print(f"step {step} | NON-FINITE LOSS - stopping. Resume from {latest_path}.")
            save(latest_path, step)
            raise SystemExit(1)

        if step % 100 == 0:
            rate = tokens_done / max(time.time() - started, 1)
            remaining = (total_steps - step) * BATCH_SIZE * config.context_length / max(rate, 1)
            vram = (f" | vram {torch.cuda.max_memory_reserved()/1024**3:.2f}G"
                    if device == "cuda" else "")
            print(f"step {step:,}/{total_steps:,} | loss {train_loss:.4f} | "
                  f"lr {lr_at(step, total_steps, args.lr, warmup):.2e} | "
                  f"{rate:,.0f} tok/s{vram} | ~{remaining/3600:.1f}h left")

        if step % EVAL == 0 or step == total_steps - 1:
            value = validation_loss()
            history.append({"step": step, "val": round(value, 4),
                            "train": round(train_loss, 4)})
            if value < best_val:
                best_val = value
                stale = 0
                save(best_path, step)
                print(f"step {step:,} | validation {value:.4f} (best)")
            else:
                stale += 1
                print(f"step {step:,} | validation {value:.4f} "
                      f"| no improvement {stale}/{PATIENCE_EVALS}")
            log_path.write_text(json.dumps(
                {"history": history, "best_val": best_val, "fingerprint": fingerprint},
                indent=2), encoding="utf-8")
            if stale >= PATIENCE_EVALS:
                print(f"early stop: {PATIENCE_EVALS} evals "
                      f"({PATIENCE_EVALS * EVAL:,} steps) without improvement")
                save(latest_path, step)
                break

        if step % CHECKPOINT_AT == 0 and step > start_step:
            save(latest_path, step)

    hours = (time.time() - started) / 3600
    print(f"\ndone in {hours:.1f}h | best validation {best_val:.4f}")
    print(f"best checkpoint {best_path}")
    report("(end of fine-tune)")


if __name__ == "__main__":
    main()
