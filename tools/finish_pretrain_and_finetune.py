"""End-of-pretrain handoff: gate, then fine-tune, then measure.

Five separate commands have to run in the right order when the pretrain finishes, and the gates
are the kind of thing that gets skipped when a run has just completed and the fine-tune is right
there. This makes the order automatic and the gates unskippable.

  1  confirm the pretrain actually finished (not a crash, not a partial run)
  2  GATE  repetition probe - strong-domain looping must stay below the tripwire
  3  GATE  packed fine-tune data verification (masking, alignment, no NaN rows)
  4  GATE  fine-tune setup verification (y from labels, causal shift, ignore_index)
  5  record the pretrained behavioural baseline for later comparison
  6  fine-tune
  7  behavioural regression on the result, against that baseline

Any gate failing stops the chain. That is the point: onboarding section 8's most expensive lesson
is that a validation loss can look textbook while the model has collapsed, so the behavioural
numbers are recorded BEFORE and AFTER rather than trusted afterwards.

Does not start anything on its own - run it deliberately.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PYTHON = sys.executable   # NOT ".venv/Scripts/python.exe" - subprocess resolves against the OS
                          # search path, not the cwd, so a relative path raises FileNotFoundError.
                          # This bug was latent: the chain only calls subprocess AFTER the pretrain
                          # finishes, so it would have fired unattended at 3am.
LOG = Path("tests/pretrain_v2_log.json")
BEST = Path("checkpoints/pretrain_v2_best.pt")
BASELINE = Path("tests/behaviour_baseline_pretrain.txt")
AFTER = Path("tests/behaviour_after_finetune.txt")
PEEK_BEFORE = Path("tests/wide_peek_pretrain.txt")
PEEK_AFTER = Path("tests/wide_peek_finetuned.txt")
TOTAL_STEPS = 130_208
# The fine-tune runs ~3.5h unattended and resumes from its last checkpoint, so a crash costs at
# most 500 steps. Without this the whole chain would stop at the first hiccup and the GPU would
# sit idle until someone noticed.
FINETUNE_RETRIES = 2
RETRY_BACKOFF = 120


def run(description, arguments, capture_to=None, gate=False, retries=0):
    """Run a step. `retries` re-runs on failure, which is only safe for resumable steps."""
    attempt = 0
    while True:
        attempt += 1
        suffix = f"  (attempt {attempt})" if attempt > 1 else ""
        print(f"\n{'=' * 74}\n{description}{suffix}\n{'=' * 74}")
        result = subprocess.run([PYTHON, "-m"] + arguments, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        output = (result.stdout or "") + (result.stderr or "")
        tail = "\n".join(line for line in output.splitlines() if line.strip())
        print(tail[-2500:])
        if capture_to:
            capture_to.write_text(output, encoding="utf-8")
            print(f"  (saved to {capture_to})")

        if result.returncode == 0:
            return 0

        print(f"\n  -> exit {result.returncode}")
        if gate:
            print(f"\nGATE FAILED: {description}")
            print("Stopping. Fix this before fine-tuning - a gate exists because the failure it")
            print("catches is silent, and would cost the whole fine-tune to discover later.")
            sys.exit(1)
        if attempt <= retries:
            # Only for steps that resume from a checkpoint. Re-running a non-resumable step
            # would silently start it over from scratch.
            print(f"  retrying in {RETRY_BACKOFF}s - the fine-tune resumes from its last "
                  f"checkpoint, so a restart costs at most 500 steps.")
            time.sleep(RETRY_BACKOFF)
            continue
        return result.returncode


def pretrain_finished():
    if not Path("tests/smoke_results/pretrain_v2_log.json").exists():
        return False, "no pretrain log"
    data = json.loads(Path("tests/smoke_results/pretrain_v2_log.json")
                      .read_text(encoding="utf-8"))
    history = data.get("history", [])
    if not history:
        return False, "log has no history"
    last = history[-1]["step"]
    if last < TOTAL_STEPS - 2_000:
        return False, f"last eval at step {last:,} of {TOTAL_STEPS:,} - still running"
    return True, f"finished at step {last:,}, best validation {data.get('best_val')}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true",
                        help="poll until the pretrain finishes instead of exiting")
    parser.add_argument("--skip-finetune", action="store_true",
                        help="run the gates and record the baseline, but do not train")
    args = parser.parse_args()

    while True:
        done, reason = pretrain_finished()
        print(f"pretrain: {reason}")
        if done:
            break
        if not args.wait:
            print("\nNot finished. Re-run with --wait to poll, or wait and run again.")
            return 1
        time.sleep(600)

    if not BEST.exists():
        print(f"missing {BEST}")
        return 1

    run("GATE 1/3  repetition probe (strong-domain looping under the tripwire)",
        ["tools.repetition_probe", "--checkpoint", str(BEST)], gate=True)
    run("GATE 2/3  packed fine-tune data",
        ["tests.verify_packed_finetune", "train"], gate=True)
    run("GATE 3/3  fine-tune setup (masking, causal shift, ignore_index)",
        ["tests.verify_finetune_v2_setup"], gate=True)

    run("BASELINE  behavioural regression on the PRETRAINED model",
        ["tests.behaviour_regression_v2", "--checkpoint", str(BEST)],
        capture_to=BASELINE)

    # Captured before AND after because the everyday-knowledge group is the one the fine-tune
    # should move most - the pretrained model cannot answer "cats have four ..." at all, and the
    # mix contains commonsenseqa, piqa and sciq. Without a before, that gain is unmeasurable.
    run("BASELINE  wide peek on the PRETRAINED model",
        ["tools.wide_peek", "--checkpoint", str(BEST)], capture_to=PEEK_BEFORE)

    if args.skip_finetune:
        print("\n--skip-finetune set: gates passed and baseline recorded. Start the fine-tune "
              "with:\n  ./.venv/Scripts/python.exe -m training.finetune_v2")
        return 0

    code = run("FINE-TUNE", ["training.finetune_v2"], retries=FINETUNE_RETRIES)
    if code != 0:
        print(f"\nFine-tune failed after {FINETUNE_RETRIES + 1} attempts. It is resumable - "
              f"re-run tools.finish_pretrain_and_finetune once the cause is fixed.")
        print(f"Baselines are already captured in {BASELINE} and {PEEK_BEFORE}.")
        return 1

    run("RESULT  behavioural regression on the FINE-TUNED model",
        ["tests.behaviour_regression_v2"], capture_to=AFTER)

    run("RESULT  wide peek on the FINE-TUNED model",
        ["tools.wide_peek", "--checkpoint", "checkpoints/finetune_v2_best.pt"],
        capture_to=PEEK_AFTER)

    print(f"\n{'=' * 74}")
    print("Compare the two behavioural scores, NOT the validation losses:")
    print(f"  before  {BASELINE}")
    print(f"  after   {AFTER}")
    print("Section 8: a validation loss can fall beautifully while the model refuses every")
    print("question. The behavioural delta is the number that means something.")
    print(f"\nAnd read the wide peeks side by side - the everyday-knowledge group is the one")
    print(f"the fine-tune should move most:")
    print(f"  before  {PEEK_BEFORE}")
    print(f"  after   {PEEK_AFTER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
