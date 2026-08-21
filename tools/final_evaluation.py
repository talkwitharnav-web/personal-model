"""Run every evaluation against the finished fine-tune and compare it to the earlier runs.

Three fine-tunes exist by the end of 2026-08-19, and comparing them is the only way to say whether
the day's data work actually bought anything:

  checkpoints/archive_shortcut_run/   trained on the yes/no set with the lexical shortcut, killed
                                      at step 10,000 when the GPU driver wedged
  checkpoints/archive_notypo_run/     shortcut fixed, no messy-input data, best 1.5084, stopped at
                                      step 32,200 to add typo augmentation
  checkpoints/finetune_v2_best.pt     the current run: shortcut fixed AND typo_aug in the mix

⚠ The three are NOT directly comparable on every axis, and the report says so rather than quietly
implying otherwise. They ran to different step counts on different data, so a difference of one or
two items on a four-item subscale is noise, not a finding. What IS comparable: the held-out yes/no
eval (same held-out file, same protocol) and the typo probe (same fixed prompts).

Everything here runs on CPU so it cannot disturb a training run that still holds the GPU.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PYTHON = sys.executable                      # absolute: a relative path resolves against PATH
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests"

JOBS = [
    ("behavioural regression (37 tests)", ["-m", "tests.behaviour_regression_v2"],
     "behaviour_FINAL.txt"),
    ("yes/no held-out accuracy", ["-m", "tools.yesno_heldout_eval", "--limit", "400"],
     "yesno_heldout_FINAL.txt"),
    ("RAG vs bare", ["-m", "tools.rag_vs_bare_peek"], "rag_vs_bare_FINAL.txt"),
    ("typo and shorthand robustness", ["-m", "tools.typo_robustness_probe"],
     "typo_probe_FINAL.txt"),
    ("fresh questions and multi-turn", ["-m", "tools.fresh_peek"], "fresh_peek_FINAL.txt"),
    # Added 2026-08-20: these three target exactly what the run's new datasets are meant to fix,
    # so they are the ones that will show whether greetings/instruction_shapes/ProofWriter worked.
    ("greetings and identity", ["-m", "tools.greeting_probe", "--samples", "4"],
     "greeting_probe_FINAL.txt"),
    ("QA stress: negation, false premises, format", ["-m", "tools.stress_qa", "--samples", "5"],
     "stress_qa_FINAL.txt"),
    ("robustness: caps, symbols, empty input", ["-m", "tools.stress_robustness",
                                                "--samples", "3"],
     "stress_robustness_FINAL.txt"),
]


def run(label, arguments, destination, checkpoint, device):
    command = [PYTHON] + arguments + ["--checkpoint", checkpoint, "--device", device]
    print(f"\n{'=' * 92}\n{label}\n{'=' * 92}", flush=True)
    try:
        result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=7200)
    except Exception as error:
        print(f"  FAILED to run ({error})")
        return None
    # Some tools write their report to stderr (anything using logging, or a tool that reconfigures
    # stdout). Keeping only stdout silently produced a 0-byte yesno_heldout_FINAL.txt while the
    # numbers appeared on screen - the report looked fine and the artifact was empty.
    text = (result.stdout or "") + (result.stderr or "")
    (OUT / destination).write_text(text, encoding="utf-8")
    if not text.strip():
        print(f"  exited {result.returncode} with NO OUTPUT AT ALL")
        return None
    print(f"  -> {OUT / destination} ({len(text):,} bytes)")
    return text


def summarise(name, text):
    """Pull the one or two lines that matter out of each report."""
    if not text:
        return
    lines = [line.rstrip() for line in text.splitlines()]
    if name == "behaviour_FINAL.txt":
        for line in lines:
            if "OVERALL" in line:
                print(f"    {line.strip()}")
    elif name == "yesno_heldout_FINAL.txt":
        for line in lines:
            if any(key in line for key in ("WITH context", "NO context", "overall", "answered Yes")):
                print(f"    {line.strip()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/finetune_v2_best.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not (ROOT / args.checkpoint).exists():
        raise SystemExit(f"missing {args.checkpoint}")

    print(f"evaluating {args.checkpoint} on {args.device}")
    collected = {}
    for label, arguments, destination in JOBS:
        collected[destination] = run(label, arguments, destination, args.checkpoint, args.device)

    print(f"\n{'=' * 92}\nHEADLINE NUMBERS\n{'=' * 92}")
    for destination in ("behaviour_FINAL.txt", "yesno_heldout_FINAL.txt"):
        summarise(destination, collected.get(destination))

    print(f"\n{'=' * 92}\nFOR COMPARISON (measured earlier today)\n{'=' * 92}")
    print("  pretrained, no fine-tune            behaviour 11/37 (30%)")
    print("  shortcut run, step 10,000           behaviour 28/37 (76%) | yes/no held-out 65.0%")
    print("  shortcut FIXED, step 27,500         behaviour 26/37 (70%) | yes/no held-out 83.2%")
    print("\n  ⚠ Those ran to different step counts on different data. The behavioural subscales")
    print("    have 2-8 items each, so a one-item change moves the total by ~3% and means nothing")
    print("    on its own. The yes/no held-out number is the comparable one: same file, same")
    print("    protocol, 400 examples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
