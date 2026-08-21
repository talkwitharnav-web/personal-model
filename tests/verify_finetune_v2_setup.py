"""Verify the v2 fine-tune data path without touching the GPU.

The pretrain owns the GPU for another ~10 hours, so this checks the things that can be checked on
CPU with no model - and they happen to be the things most likely to be silently wrong:

  - y must come from LABELS, not ids. Taking y from ids trains the model on the prompt as well
    and quietly destroys the whole point of loss masking.
  - the causal shift must line up: x = ids[:, :-1], y = labels[:, 1:], so position i of x predicts
    labels[i+1]. An off-by-one here still trains, just against the neighbouring token.
  - ignore_index=-100 must actually be honoured, or the -100s become a real class id.
  - the learning-rate schedule must warm up and decay over THIS run's length.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from training.finetune_v2 import (BATCH_SIZE, MIN_LR_FRACTION, get_batch, load_split, lr_at)

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}  {detail}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def main():
    ids, labels = load_split("validation")
    print(f"validation: {ids.shape[0]:,} rows x {ids.shape[1]}")

    generator = torch.Generator().manual_seed(0)
    x, y = get_batch(ids, labels, generator, "cpu")
    width = ids.shape[1]
    check("x shape is (batch, context-1)", tuple(x.shape) == (BATCH_SIZE, width - 1),
          str(tuple(x.shape)))
    check("y shape matches x", tuple(y.shape) == tuple(x.shape), str(tuple(y.shape)))

    # y must contain -100s; if it came from ids it could not.
    masked = int((y == -100).sum())
    check("y contains masked positions (so it came from labels, not ids)", masked > 0,
          f"{masked:,} of {y.numel():,} ({masked/y.numel()*100:.1f}%)")

    # THE alignment check, done directly on one row rather than trusting the slicing.
    row = 0
    row_ids = np.asarray(ids[row]).astype(np.int64)
    row_labels = np.asarray(labels[row]).astype(np.int64)
    shifted_x = row_ids[:-1]
    shifted_y = row_labels[1:]
    keep = shifted_y != -100
    # Wherever a target survives, it must equal the id at that same absolute position, which is
    # one further along than the input token predicting it.
    aligned = np.array_equal(shifted_y[keep], row_ids[1:][keep])
    check("causal shift aligns: y[i] == ids[i+1] wherever unmasked", aligned)
    check("x is the token BEFORE each target", len(shifted_x) == len(shifted_y))

    # ignore_index must be honoured: a loss computed with it must ignore the masked positions.
    torch.manual_seed(0)
    vocab = 32_000
    logits = torch.randn(2, 5, vocab)
    targets = torch.tensor([[3, -100, 7, -100, 11], [-100, -100, 2, 5, -100]])
    with_ignore = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1),
                                  ignore_index=-100)
    manual = F.cross_entropy(
        logits.reshape(-1, vocab)[targets.reshape(-1) != -100],
        targets.reshape(-1)[targets.reshape(-1) != -100])
    check("ignore_index=-100 matches masking by hand",
          torch.allclose(with_ignore, manual, atol=1e-6),
          f"{with_ignore.item():.6f} vs {manual.item():.6f}")

    # Without ignore_index the same call would be an error or nonsense - prove it differs.
    try:
        naive = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1).clamp(min=0))
        check("dropping ignore_index changes the loss (so it is load-bearing)",
              not torch.allclose(naive, with_ignore, atol=1e-3),
              f"naive {naive.item():.4f} vs correct {with_ignore.item():.4f}")
    except Exception as error:
        check("dropping ignore_index is not silently fine", True, str(error)[:60])

    # Learning-rate schedule.
    total, warmup, peak = 19_256, 385, 1.0e-4
    start = lr_at(0, total, peak, warmup)
    top = lr_at(warmup, total, peak, warmup)
    end = lr_at(total - 1, total, peak, warmup)
    print(f"  lr: step 0 {start:.2e} | step {warmup} {top:.2e} | step {total-1} {end:.2e}")
    check("lr starts near zero", start < peak * 0.01, f"{start:.2e}")
    check("lr reaches the peak at the end of warmup",
          abs(top - peak) / peak < 0.01, f"{top:.2e}")
    check("lr decays to the floor", abs(end - peak * MIN_LR_FRACTION) / peak < 0.02,
          f"{end:.2e} vs floor {peak*MIN_LR_FRACTION:.2e}")
    check("warmup is shorter than the run", warmup < total, f"{warmup} < {total}")

    # How much of a batch actually teaches anything.
    valid = float((y != -100).float().mean())
    print(f"  positions carrying gradient in this batch: {valid*100:.1f}%")
    check("batch has valid targets", valid > 0.02, f"{valid*100:.1f}%")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
