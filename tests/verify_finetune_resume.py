"""Prove the fine-tune will actually RESUME from the checkpoint on disk - without a GPU.

The restart after the crash never got far enough to print "resumed from ...": it hung in CUDA init
because the driver was wedged. So the resume path is still unproven, and it is now the only path
that matters - the next start Dark makes is a resume, not a fresh run.

A resume can fail for a reason that has nothing to do with the GPU: `finetune_v2.py` refuses to
load a checkpoint whose fingerprint differs from the current configuration, and the configuration
was edited after the run started (BATCH_SIZE 16 -> 8, GRAD_ACCUM 4 -> 8, EVAL_EVERY 250 -> 500).
If those constants no longer match what is baked into the checkpoint, the next start exits with
"was written by a DIFFERENT configuration" and Dark wakes up to a run that never restarted.

Everything here is CPU-only and read-only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

from training import finetune_v2 as ft

passed, failed = 0, 0


def check(label, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {label}  {detail}")
    else:
        failed += 1
        print(f"  FAIL  {label}  {detail}")


def main():
    latest = ft.CHECKPOINT_DIR / "finetune_v2_latest.pt"
    print(f"=== resume check: {latest} ===")
    if not latest.exists():
        print("  no checkpoint - nothing to resume from (a fresh start is expected)")
        return 0

    train_ids = np.load(ft.DATA_DIR / "train_ids.npy", mmap_mode="r")

    # Recompute the fingerprint exactly as main() does, from the CURRENT constants.
    epochs, lr = 2.0, ft.PEAK_LR
    steps_per_epoch = len(train_ids) // ft.BATCH_SIZE
    total_steps = int(epochs * steps_per_epoch)
    warmup = max(50, int(total_steps * ft.WARMUP_FRACTION))
    expected = {"epochs": epochs, "lr": lr, "warmup": warmup, "batch": ft.BATCH_SIZE,
                "grad_accum": ft.GRAD_ACCUM, "rows": int(len(train_ids)), "arch": "v2"}

    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    saved = checkpoint.get("fingerprint", {})

    print(f"  saved    {saved}")
    print(f"  current  {expected}")
    differences = {k: (saved.get(k), expected[k]) for k in expected if saved.get(k) != expected[k]}
    check("fingerprint matches, so the resume will not be refused",
          not differences, str(differences) if differences else "identical")

    step = checkpoint.get("step", -1)
    check("checkpoint carries a step to resume from", step > 0, f"resumes at step {step + 1:,}")
    check("checkpoint is not already past the end", step < total_steps,
          f"{step:,} of {total_steps:,}")

    for key in ("model", "optimizer", "scaler", "best_val", "history"):
        check(f"checkpoint contains '{key}'", key in checkpoint)

    # The state dict must actually fit the model this code builds now.
    config = ft.v2_config()
    model = ft.DarkGLM(config)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    check("model state_dict loads with no missing keys", not missing, str(list(missing)[:3]))
    check("model state_dict loads with no unexpected keys", not unexpected,
          str(list(unexpected)[:3]))

    best = checkpoint.get("best_val", float("nan"))
    check("best validation is a real improvement over the pretrain (2.7667)",
          best < 2.7667, f"{best:.4f}")

    history = checkpoint.get("history", [])
    if history:
        rising = sum(1 for a, b in zip(history, history[1:]) if b["val"] > a["val"])
        print(f"  {len(history)} evals recorded, {rising} rose "
              f"(first {history[0]['val']:.4f} -> last {history[-1]['val']:.4f})")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
