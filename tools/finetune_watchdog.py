"""Keep the v2 fine-tune alive on a SHARED machine.

Why this exists, specifically
-----------------------------
The fine-tune died at step 9,100 with `CUBLAS_STATUS_EXECUTION_FAILED` - not because of anything
in the training config, but because another user's parallel C++ build (16 concurrent `cl.exe`
processes) took the host RAM down to 91 MB. CUDA needs host-side staging memory, so host pressure
surfaces as a *GPU* error, which is thoroughly misleading. Onboarding section 2 already records
Python dying silently under memory pressure; this is the same trap wearing a cuBLAS costume.

The pretrain's watchdog (`tools/train_watchdog.py`) is hardcoded to `pretrain_v2` and, more
importantly, restarts on death alone. That is not enough here: restarting into a machine that is
still out of RAM just burns a restart and dies again. This one gates on free RAM before it starts
anything, and treats "no RAM yet" as a wait rather than a failure.

Resume is automatic: `finetune_v2.py` reloads `checkpoints/finetune_v2_latest.pt` when it exists
and the fingerprint matches, so a restart costs at most CHECKPOINT_EVERY steps of work.
"""
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tools.memtrack import system_free_mb

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)          # absolute: a restart must not depend on the cwd
LOG_DIR = ROOT / "logs"
CONSOLE_LOG = LOG_DIR / "finetune_v2.log"
ERROR_LOG = LOG_DIR / "finetune_v2.err"
WATCHDOG_LOG = ROOT / "tests" / "finetune_watchdog.log"

POLL_SECONDS = 120
# The trainer needs ~1.7 GB working set. Starting it with less than this free on a machine that
# is actively being eaten by someone else's build is how the last run died.
MIN_FREE_TO_START_MB = 2600
MAX_RESTARTS = 25                      # shared machine: be persistent, not proud
TOTAL_STEPS = 42_966
# A start that dies before reaching a checkpoint makes no progress. If that happens repeatedly the
# problem is not transient and a human needs to look.
MAX_STARTS_WITHOUT_PROGRESS = 4
# A HUNG trainer looks exactly like a healthy one to a process check, and that is not theoretical:
# after the cuBLAS failure the GPU driver wedged, and the next start sat in CUDA init forever
# having burned 3.8 s of CPU in ten minutes. Without this the watchdog waits for a corpse.
# The trainer prints every 100 steps (~55 s) and resume costs a 911 MB checkpoint load, so 15
# minutes of no new step is well past anything legitimate.
SILENCE_LIMIT = 900


def note(message):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(WATCHDOG_LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_text(path):
    """Read a console log whatever encoding PowerShell chose for it (section 13)."""
    if not path.exists():
        return ""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def gpu_healthy(timeout=60):
    """True if the GPU driver answers at all.

    After the cuBLAS failure, `nvidia-smi` itself hung indefinitely and the trainer process could
    not be killed - it sat in a kernel wait on the driver. Restarting into that state is pointless:
    CUDA init never returns, so the new process hangs too and looks alive to a process check.
    A driver that cannot answer a status query in a minute is not going to run a training step.

    ⚠ This deliberately does NOT use `subprocess.run(timeout=...)`. That path kills the child and
    then *waits* for it, and a process wedged in the GPU driver cannot be reaped - so the call that
    exists to bound the wait blocks forever. Measured: a 45 s timeout took over 6 minutes and was
    still going. Poll, then abandon the handle.
    """
    try:
        process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            try:
                out = process.stdout.read() if process.stdout else ""
            except Exception:
                out = ""
            return process.returncode == 0 and bool(out.strip())
        time.sleep(2)

    try:                       # best effort only - never wait() on it
        process.kill()
    except Exception:
        pass
    return False


def trainer_pids():
    """PIDs of every python process running training.finetune_v2 (trampoline included)."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*training.finetune_v2*' } | "
             "Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=120)
        return [int(line) for line in (result.stdout or "").split() if line.strip().isdigit()]
    except Exception as error:
        note(f"pid lookup failed ({error})")
        return []


def trainer_running():
    """True if a python process is running training.finetune_v2.

    Matches the module path exactly so that neither this watchdog nor the peek monitor - whose
    command lines also contain "finetune" - can be mistaken for the trainer.
    """
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*training.finetune_v2*' } | Measure-Object).Count"],
            capture_output=True, text=True, timeout=120)
        return int((result.stdout or "0").strip() or 0) > 0
    except Exception as error:                 # never let the check itself kill the watchdog
        note(f"process check failed ({error}) - assuming alive, will not restart")
        return True


def kill_trainer():
    """Kill a hung trainer by PID. A process wedged in the GPU driver may refuse to die."""
    for pid in trainer_pids():
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"],
                           capture_output=True, timeout=60)
            note(f"  sent kill to PID {pid}")
        except Exception as error:
            note(f"  kill of PID {pid} failed ({error})")


def progress():
    """(last step seen, finished?) across every log part this run has produced."""
    text = read_text(CONSOLE_LOG)
    for older in sorted(LOG_DIR.glob("finetune_v2.part*.log")):
        text += read_text(older)
    finished = "best checkpoint" in text and "done in" in text
    steps = re.findall(r"^step ([\d,]+)/", text, flags=re.MULTILINE)
    last = max((int(s.replace(",", "")) for s in steps), default=0)
    return last, finished


def rotate_logs():
    """Keep the failed run's output instead of letting the next start overwrite it."""
    if not CONSOLE_LOG.exists():
        return
    index = 1
    while (LOG_DIR / f"finetune_v2.part{index}.log").exists():
        index += 1
    CONSOLE_LOG.rename(LOG_DIR / f"finetune_v2.part{index}.log")
    if ERROR_LOG.exists():
        ERROR_LOG.rename(LOG_DIR / f"finetune_v2.part{index}.err")


def start_trainer():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rotate_logs()
    out = open(CONSOLE_LOG, "w", encoding="utf-8")
    err = open(ERROR_LOG, "w", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), "-u", "-m", "training.finetune_v2"],
        cwd=str(ROOT), stdout=out, stderr=err,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return process.pid


def run_final_tests():
    """The tests Dark asked for once the fine-tune finishes.

    Run from here rather than by hand, because the run now finishes after a reboot - possibly with
    nobody watching. Results land next to the pretrained baselines so the two can be diffed:
      tests/behaviour_baseline_pretrain.txt  vs  tests/behaviour_finetuned.txt
      tests/wide_peek_pretrain.txt           vs  tests/wide_peek_finetuned.txt
    """
    jobs = [
        ("behavioural regression",
         [str(PYTHON), "-m", "tests.behaviour_regression_v2",
          "--checkpoint", "checkpoints/finetune_v2_best.pt"],
         ROOT / "tests" / "behaviour_finetuned.txt"),
        ("wide peek",
         [str(PYTHON), "-m", "tools.wide_peek",
          "--checkpoint", "checkpoints/finetune_v2_best.pt"],
         ROOT / "tests" / "wide_peek_finetuned.txt"),
    ]
    for label, command, destination in jobs:
        note(f"running {label} on the fine-tuned checkpoint ...")
        try:
            result = subprocess.run(command, cwd=str(ROOT), capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=3600)
            destination.write_text(result.stdout or "", encoding="utf-8")
            tail = [line for line in (result.stdout or "").splitlines() if line.strip()][-6:]
            for line in tail:
                note(f"  {line}")
            note(f"  -> saved to {destination}")
        except Exception as error:
            note(f"  {label} FAILED ({error}) - run it by hand")


def main():
    note(f"fine-tune watchdog started | poll {POLL_SECONDS}s | "
         f"needs {MIN_FREE_TO_START_MB:,} MB free to (re)start | max {MAX_RESTARTS} restarts")
    starts = 0
    step_at_last_start = -1
    starts_without_progress = 0
    seen_step = None
    step_changed_at = time.time()

    while True:
        last_step, finished = progress()

        if last_step != seen_step:
            seen_step = last_step
            step_changed_at = time.time()

        if finished:
            note(f"FINE-TUNE COMPLETED at step {last_step:,}. Exiting cleanly.")
            tail = [line for line in read_text(CONSOLE_LOG).splitlines() if line.strip()][-3:]
            for line in tail:
                note(f"  {line}")
            if "--no-test" not in sys.argv:
                run_final_tests()
            return 0

        if trainer_running():
            silent_for = time.time() - step_changed_at
            if silent_for > SILENCE_LIMIT:
                note(f"trainer process is alive but has logged no new step in "
                     f"{silent_for/60:.0f} min (stuck at {last_step:,}) - treating as HUNG")
                if not gpu_healthy():
                    note("  the GPU driver is not answering either. The process is wedged in the "
                         "driver and cannot be killed; this needs a reboot, not a restart. "
                         "Waiting - the checkpoint on disk is safe.")
                else:
                    kill_trainer()
                step_changed_at = time.time()   # do not re-kill on the very next poll
            time.sleep(POLL_SECONDS)
            continue

        # Not running and not finished -> it died.
        if last_step > step_at_last_start:
            starts_without_progress = 0        # it did move forward, so the restart was worth it
        elif starts > 0:
            starts_without_progress += 1

        if starts_without_progress >= MAX_STARTS_WITHOUT_PROGRESS:
            note(f"STOPPING: {starts_without_progress} starts in a row made no progress past "
                 f"step {last_step:,}. This is not transient - a human needs to look at "
                 f"{ERROR_LOG}.")
            return 1

        if starts >= MAX_RESTARTS:
            note(f"STOPPING: hit the {MAX_RESTARTS}-restart limit at step {last_step:,}.")
            return 1

        free = system_free_mb()
        if free is not None and free < MIN_FREE_TO_START_MB:
            note(f"trainer is down at step {last_step:,}, but only {free:,.0f} MB free "
                 f"(need {MIN_FREE_TO_START_MB:,}) - waiting rather than restarting into an OOM")
            time.sleep(POLL_SECONDS)
            continue

        if not gpu_healthy():
            note(f"trainer is down at step {last_step:,} and the GPU driver is not answering - "
                 f"waiting for it to recover rather than starting a process that will hang")
            time.sleep(POLL_SECONDS)
            continue

        starts += 1
        step_at_last_start = last_step
        pid = start_trainer()
        note(f"restart {starts}/{MAX_RESTARTS}: trainer was down at step {last_step:,}, "
             f"{free:,.0f} MB free -> started PID {pid} (resumes from the latest checkpoint)")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
