"""Final check on the combined fine-tune dataset, immediately before training on it.

Everything here has been verified at some point, but not all at once and not all after the last
regeneration. definitions, gsm8k, proofwriter and yesno were all rebuilt today, and the pack was
redone four times. This is the "look at it once more before you spend hours" pass.

Checks, each tied to a documented failure:
  shape        rows/width match, ids and labels aligned (section 5)
  masking      no row entirely -100 (NaN loss), unmasked labels equal ids (off-by-one)
  content      decode real rows and print them - structure proves nothing about quality
  balance      label distribution for the classification-style sets
  shortcuts    majority-class and context-reuse, the BoolQ defect (section 9)
  diversity    question phrasing spread, the defect fixed today (section 6)
  markers      every dataset still produces a \\nAssistant: marker after retokenizing
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# sys.executable, not a relative path: subprocess resolves against the OS search path, not the
# working directory, so ".venv/Scripts/python.exe" raises FileNotFoundError. This is also the
# interpreter already running, which guarantees the child uses the same venv.
PYTHON = sys.executable
PACKED = Path("data/finetune_packed")
TOKENIZER = Path("tokenizer/tokenizer_v2.json")

failures = []


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def run(description, arguments):
    print(f"\n--- {description}")
    result = subprocess.run([PYTHON, "-m"] + arguments, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    output = ((result.stdout or "") + (result.stderr or "")).strip()
    print("\n".join(output.splitlines()[-14:]))
    if result.returncode != 0:
        failures.append(description)
    return result.returncode


def main():
    section("1. PACKED DATA - shape, masking, alignment")
    run("packed train split", ["tests.verify_packed_finetune", "train"])
    run("fine-tune data path (y from labels, causal shift, ignore_index)",
        ["tests.verify_finetune_v2_setup"])

    section("2. WHAT THE MODEL WILL ACTUALLY SEE")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    ids = np.load(PACKED / "train_ids.npy", mmap_mode="r")
    labels = np.load(PACKED / "train_labels.npy", mmap_mode="r")
    rng = np.random.default_rng(11)
    for row in rng.choice(len(ids), 3, replace=False):
        row_ids = np.asarray(ids[row])
        row_labels = np.asarray(labels[row])
        trained = tokenizer.decode([int(t) for t in row_ids[row_labels != -100]])
        full = tokenizer.decode([int(t) for t in row_ids])
        print(f"\n  row {row:,}")
        print(f"    full     {full[:220].replace(chr(10), ' | ')!r}")
        print(f"    TRAINED  {trained[:220].replace(chr(10), ' | ')!r}")
        print(f"    gradient on {float((row_labels != -100).mean())*100:.1f}% of positions")

    section("3. SHORTCUTS AND BALANCE ACROSS EVERY DATASET")
    run("garbage and shortcut audit", ["tools.audit_finetune_datasets"])

    section("4. QUESTION PHRASING DIVERSITY")
    run("template poverty check", ["tools.audit_question_diversity"])

    section("5. THE MIX ITSELF")
    run("dataset inventory with samples", ["tools.show_finetune_mix"])

    section("VERDICT")
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for item in failures:
            print(f"  - {item}")
        print("\nDo not start the fine-tune until these are resolved.")
        return 1
    print("All checks passed. The combined dataset is ready to train on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
