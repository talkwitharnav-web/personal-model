"""Extrapolate the pretrain loss curve to its end, from the run's own validation points.

Neural loss against tokens seen follows a power law well: L(D) = A + B * D**(-alpha), where A is
the irreducible floor for this model size and data. Three unknowns, so this needs at least three
validation points and gets steadier as more arrive.

Two things this CANNOT capture, both stated in the output rather than hidden:
  - cosine decay's end-of-run drop. Annealing the LR from 1e-3 down to 1e-4 over the final stretch
    buys a real improvement that a power law fitted to the high-LR phase will not see.
  - loss is TOKENIZER-DEPENDENT. tokenizer_v2 packs 4.165 bytes per token against v1's 3.773, so
    each token carries more information and per-token loss is legitimately HIGHER for the same
    quality of model. Never compare these numbers to a run on a different tokenizer.
"""
import json
import sys
from pathlib import Path

import numpy as np

LOG = Path("tests/smoke_results/pretrain_v2_log.json")
TOKENS_PER_STEP = 16 * 768
TOTAL_STEPS = 130_208


def fit_power_law(steps, losses):
    """Grid search over (A, alpha), fitting B in closed form, and return an uncertainty band.

    Fitting three free parameters to three points gives an EXACT fit with zero degrees of
    freedom - it interpolates and tells you nothing about the future. The first version of this
    did exactly that and returned alpha = 1.42, which would mean the loss had already converged
    at step 6,000. Real data-scaling exponents are ~0.05-0.15, so alpha is constrained to a
    plausible band and the spread of near-best fits is reported as the actual uncertainty.
    """
    tokens = steps * TOKENS_PER_STEP
    log_tokens = np.log(tokens)
    candidates = []
    for alpha in np.arange(0.02, 0.31, 0.005):
        for floor in np.arange(0.0, min(losses) - 0.05, 0.02):
            residual = losses - floor
            if np.any(residual <= 0):
                continue
            # L - A = B * D^-alpha  ->  log(L-A) + alpha*log(D) = log(B), best B is the mean.
            log_b = float(np.mean(np.log(residual) + alpha * log_tokens))
            b = np.exp(log_b)
            predicted = floor + b * tokens ** (-alpha)
            error = float(np.sum((predicted - losses) ** 2))
            candidates.append((error, floor, b, alpha))
    candidates.sort()
    return candidates


def main():
    data = json.loads(LOG.read_text(encoding="utf-8"))
    history = [h for h in data["history"] if h["step"] > 0]
    if len(history) < 3:
        print(f"only {len(history)} validation points past step 0 - need at least 3.")
        for h in data["history"]:
            print(f"  step {h['step']:>7,}  val {h['val']:.4f}")
        return 0

    steps = np.array([h["step"] for h in history], dtype=float)
    losses = np.array([h["val"] for h in history], dtype=float)

    print("validation so far")
    for h in history:
        print(f"  step {h['step']:>7,}  {h['val']:.4f}")
    degrees = len(history) - 3
    print(f"\n{len(history)} points, 3 parameters -> {degrees} degrees of freedom"
          + ("  (TOO FEW - treat as indicative only)" if degrees < 2 else ""))

    candidates = fit_power_law(steps, losses)
    error, floor, b, alpha = candidates[0]
    final_tokens = TOTAL_STEPS * TOKENS_PER_STEP
    prediction = floor + b * final_tokens ** (-alpha)

    # Everything within a small multiple of the best error is an equally defensible fit.
    threshold = max(error * 3, 1e-4)
    band = [f + bb * final_tokens ** (-a) for e, f, bb, a in candidates if e <= threshold]

    print(f"\nbest fit: L(D) = {floor:.3f} + {b:.4g} * D^(-{alpha:.3f})   (sse {error:.6f})")
    print(f"  central estimate at {TOTAL_STEPS:,} steps "
          f"({final_tokens/1e9:.2f}B tokens): {prediction:.2f}")
    if band:
        print(f"  plausible range from comparable fits: {min(band):.2f} - {max(band):.2f}")

    for fraction in (0.25, 0.5, 1.0):
        step = int(TOTAL_STEPS * fraction)
        value = floor + b * (step * TOKENS_PER_STEP) ** (-alpha)
        print(f"    at {fraction*100:>3.0f}% ({step:>7,} steps): {value:.2f}")

    print("\ncaveats")
    print("  - cosine decay to 1e-4 over the final stretch buys a real extra drop that a power")
    print("    law fitted to the high-LR phase does not model. Expect to beat this estimate.")
    print("  - 15% of the corpus is TinyStories, which is much easier than Wikipedia and pulls")
    print("    the average down; this number is not a pure encyclopaedic-text loss.")
    print("  - loss is TOKENIZER-DEPENDENT. tokenizer_v2 packs 4.165 bytes/token vs v1's 3.773,")
    print("    so each token carries more information and per-token loss is legitimately higher")
    print("    for the same quality. Never compare across tokenizers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
