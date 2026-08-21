"""Wait for the fine-tune to finish, then run every evaluation against the new checkpoint.

Dark went to sleep with the run started and asked for it to be managed. The run is ~5 hours and
the evaluation suite is another ~20 minutes, so chaining them means the results are waiting rather
than needing someone awake at the handover.

Waits for `tools/finetune_watchdog.py` to exit - it already knows how to tell a clean completion
from a crash, restarts the trainer when the machine allows, and gives up only after repeated
no-progress starts. When it exits 0 the run genuinely finished; anything else means it stopped for
a reason a human needs to see, so no evaluation is run.

⚠ Deliberately does NOT switch anything over or delete anything. `checkpoints/finetune_v2_best.pt`
is first in CHECKPOINT_CANDIDATES, so it becomes what chat.py serves the moment it exists - that is
already true and is not this script's doing. The three previous runs stay archived under
`checkpoints/archive_*` for comparison.
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
LOG = ROOT / "tests" / "post_training_chain.log"
EVAL_LOG = ROOT / "logs" / "final_evaluation.log"
CHECKPOINT = ROOT / "checkpoints" / "finetune_v2_best.pt"

POLL_SECONDS = 180


def note(message):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def count(pattern):
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             f"Where-Object {{ $_.CommandLine -like '*{pattern}*' }} | Measure-Object).Count"],
            capture_output=True, text=True, timeout=120)
        return int((result.stdout or "0").strip() or 0)
    except Exception as error:
        note(f"process check failed ({error}) - assuming still running")
        return 1


def finished_cleanly():
    """True if the trainer printed its completion line."""
    log = ROOT / "logs" / "finetune_v2.log"
    if not log.exists():
        return False
    text = log.read_bytes().decode("utf-8", errors="replace")
    return "done in" in text and "best checkpoint" in text


def main():
    note("waiting for the fine-tune to finish before running the evaluation suite")
    while True:
        if count("finetune_watchdog") == 0 and count("training.finetune_v2") == 0:
            break
        time.sleep(POLL_SECONDS)

    if not finished_cleanly():
        note("the watchdog exited but the trainer never printed a completion line - it gave up "
             "rather than finished. NOT running the evaluation; see tests/finetune_watchdog.log")
        return 1

    log = ROOT / "logs" / "finetune_v2.log"
    tail = [line for line in log.read_bytes().decode("utf-8", errors="replace").splitlines()
            if line.strip()][-3:]
    for line in tail:
        note(f"  {line}")

    if not CHECKPOINT.exists():
        note(f"missing {CHECKPOINT} - nothing to evaluate")
        return 1

    note("running the full evaluation suite (8 tools) - this takes about 20 minutes")
    EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_LOG, "w", encoding="utf-8") as out:
        code = subprocess.run(
            [str(PYTHON), "-u", "-m", "tools.final_evaluation",
             "--checkpoint", "checkpoints/finetune_v2_best.pt", "--device", "cuda"],
            cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT, timeout=10800).returncode
    note(f"evaluation suite exited {code} - results in tests/*_FINAL.txt, console in {EVAL_LOG}")

    for line in [l for l in EVAL_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
                 if "OVERALL" in l or "overall:" in l or "WITH context" in l
                 or "NO context" in l or "mean" in l][:12]:
        note(f"  {line.strip()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
