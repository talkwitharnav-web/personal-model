"""Smoke test: v1 architecture vs v2, everything else held identical.

Answers "is the new architecture actually better" for ~90 minutes instead of finding out
29 hours into a pretrain. Trains from RANDOM INIT on the flat Wikipedia token stream - the
fine-tune .npy rows average 220 real tokens of 768, which would hide the whole point of RoPE.

  python -m training.smoke_test --arch v1
  python -m training.smoke_test --arch v2
  python -m training.smoke_test --arch v1 --steps 800 --lr 9e-4 --warmup 100 --eval-every 50 --tag stress   (test C)

Test C note: --warmup MUST be shorter than --steps or the run never reaches the peak LR and
the stress is never actually applied - both arms then return "spikes 0", which reads as a
pass when nothing was tested.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.config import DarkGLMConfig, v2_config
from model.model import DarkGLM

BATCH_SIZE = 16
GRAD_ACCUM = 4
EVAL_EVERY = 250
EVAL_BATCHES = 20
WARMUP = 500
INIT_SEED = 1234        # model init - fixed so a rerun reproduces
DATA_SEED = 99          # batch order - MUST be independent of init (see get_batch)
VAL_SEED = 7
RESULTS_DIR = Path("tests/smoke_results")

# Measured on data/wikipedia/validation.bin: the cross-entropy of a model that has learned only
# how often each token occurs and nothing whatsoever about context (a unigram model). This - not
# ln(vocab)=10.3735 - is the real "learned nothing useful" floor. Test C found v1 at 3x LR
# finishing at 7.4386, i.e. within 0.014 of this, WITHOUT ever spiking or producing a NaN.
UNIGRAM_BASELINE = 7.4530
NEAR_UNIGRAM_MARGIN = 0.15
MIN_SECOND_HALF_GAIN = 0.25


def assess(history):
    """Pass/fail on the shape of the curve, not just 'did it crash'.

    Post-norm at 3x LR does not diverge - it silently stops improving, so every crash-based
    check passes while the model is learning nothing. Stagnation has to be tested for directly.
    """
    if len(history) < 3:
        return {"verdict": "INCONCLUSIVE", "reasons": ["not enough eval points"]}

    final = history[-1]["val"]
    midpoint = history[len(history) // 2]
    gain = midpoint["val"] - final

    reasons = []
    if gain < MIN_SECOND_HALF_GAIN:
        reasons.append(
            f"STAGNATED: validation improved only {gain:.4f} from step {midpoint['step']} "
            f"to {history[-1]['step']} (need >= {MIN_SECOND_HALF_GAIN})"
        )
    if final >= UNIGRAM_BASELINE - NEAR_UNIGRAM_MARGIN:
        reasons.append(
            f"AT UNIGRAM FLOOR: final {final:.4f} is not meaningfully below the frequency-table "
            f"baseline {UNIGRAM_BASELINE} (needs to beat it by > {NEAR_UNIGRAM_MARGIN}) - "
            f"the model is not using context"
        )
    return {
        "verdict": "FAIL" if reasons else "PASS",
        "second_half_gain": round(gain, 4),
        "vs_unigram": round(UNIGRAM_BASELINE - final, 4),
        "reasons": reasons,
    }


def build_model(arch, device):
    config = DarkGLMConfig() if arch == "v1" else v2_config()
    torch.manual_seed(INIT_SEED)
    model = DarkGLM(config).to(device)
    model.gradient_checkpointing = True
    model.train()
    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["v1", "v2"], required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tag", default="race")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--eval-every", dest="eval_every", type=int, default=EVAL_EVERY)
    args = parser.parse_args()
    if args.warmup >= args.steps:
        raise SystemExit(
            f"--warmup ({args.warmup}) >= --steps ({args.steps}): the peak lr would never be "
            f"reached and the run would prove nothing."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = build_model(args.arch, device)
    params = sum(p.numel() for p in model.parameters())

    train_data = np.memmap("data/wikipedia/train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap("data/wikipedia/validation.bin", dtype=np.uint16, mode="r")

    # A DEDICATED generator for data. Model init consumes a different amount of randomness
    # for v1 vs v2 (v2 has no position table), so sharing the global RNG would hand the two
    # arms DIFFERENT BATCHES and quietly invalidate the whole comparison.
    train_rng = torch.Generator().manual_seed(DATA_SEED)

    def get_batch(data, generator):
        top = len(data) - config.context_length - 1
        starts = torch.randint(top, (BATCH_SIZE,), generator=generator)
        x = torch.stack([torch.from_numpy(data[i:i + config.context_length].astype(np.int64))
                         for i in starts])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + config.context_length].astype(np.int64))
                         for i in starts])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def validation_loss():
        model.eval()
        # Re-seeded every time, so every eval scores on the SAME batches - removes eval
        # noise from the curve and makes the two arms directly comparable.
        val_rng = torch.Generator().manual_seed(VAL_SEED)
        total = 0.0
        for _ in range(EVAL_BATCHES):
            x, y = get_batch(val_data, val_rng)
            with torch.autocast(device_type=device, dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            total += loss.item()          # .item() - keeping the tensor would hold the graph
        model.train()
        return total / EVAL_BATCHES

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.1 + args.lr * 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.tag}_{args.arch}.json"
    history, spikes = [], 0
    train_curve, train_spikes, max_train = [], 0, 0.0
    previous_val = None

    print(f"=== SMOKE TEST {args.arch.upper()} | {args.tag} ===")
    print(f"device {device} | params {params:,} | steps {args.steps} | peak lr {args.lr:g}")
    print(f"pre_norm={config.pre_norm} use_rope={config.use_rope} "
          f"scale_residual_init={config.scale_residual_init}")

    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    start = time.time()
    tokens_done = 0

    for step in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)

        x, y = get_batch(train_data, train_rng)
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        tokens_done += BATCH_SIZE * config.context_length

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = loss.item() * GRAD_ACCUM
        if not math.isfinite(train_loss):
            print(f"Step {step} | NON-FINITE LOSS - diverged")
            spikes += 1000
            break

        # Train-loss spikes are the fastest divergence signal - a jump that recovers between
        # two validation evals is invisible to the val check below. Window is taken BEFORE
        # appending, so the current step never pollutes its own baseline.
        window = train_curve[-50:]
        if len(window) >= 20:
            baseline = sum(window) / len(window)
            if train_loss > baseline + 1.0:
                train_spikes += 1
                print(f"  TRAIN SPIKE step {step} | {baseline:.4f} -> {train_loss:.4f}")
        train_curve.append(train_loss)
        max_train = max(max_train, train_loss)

        if step % 100 == 0:
            print(f"Step {step}/{args.steps} | Loss {train_loss:.4f} | lr {lr_at(step):.2e}")

        if step % args.eval_every == 0 or step == args.steps - 1:
            value = validation_loss()
            # A spike is a jump big enough that it can't be ordinary batch noise - the
            # signature post-norm is expected to show at a high learning rate.
            if previous_val is not None and value > previous_val + 0.5:
                spikes += 1
                print(f"  SPIKE {previous_val:.4f} -> {value:.4f}")
            previous_val = value
            history.append({"step": step, "val": round(value, 4), "train": round(train_loss, 4)})
            print(f"Step {step}/{args.steps} | Validation {value:.4f}")

    elapsed = time.time() - start
    result = {
        "arch": args.arch,
        "tag": args.tag,
        "params": params,
        "steps": args.steps,
        "peak_lr": args.lr,
        "warmup": args.warmup,
        "eval_every": args.eval_every,
        "history": history,
        "train_curve": [round(v, 4) for v in train_curve],
        "best_val": min((h["val"] for h in history), default=None),
        "final_val": history[-1]["val"] if history else None,
        "spikes": spikes,
        "train_spikes": train_spikes,
        "max_train_loss": round(max_train, 4),
        "assessment": assess(history),
        "tokens_per_sec": round(tokens_done / elapsed),
        "minutes": round(elapsed / 60, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nbest {result['best_val']} | final {result['final_val']} | spikes {spikes} "
          f"| train_spikes {train_spikes} | max_train {max_train:.4f} "
          f"| {result['tokens_per_sec']:,} tok/s | {result['minutes']} min")
    verdict = result["assessment"]
    print(f"VERDICT {verdict['verdict']} | second-half gain {verdict.get('second_half_gain')} "
          f"| below unigram floor by {verdict.get('vs_unigram')}")
    for reason in verdict["reasons"]:
        print(f"  ! {reason}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
