"""Overnight watchdog for the v2 pretrain.

The run is 30 hours and resumable, so the expensive failure is not a crash - it is a crash that
goes unnoticed for six hours. This polls, and if the trainer has stopped, restarts it from the
last checkpoint.

Deliberately conservative:
  - restarts ONLY when the log has gone quiet AND no trainer process is alive, so it can never
    fight a healthy run
  - capped restarts with backoff, so a persistent failure does not become a restart loop that
    burns the GPU all night
  - never passes --fresh, so a fingerprint mismatch stays fatal rather than being "fixed" by
    silently discarding progress (onboarding section 5's resume trap)
  - logs every decision, so the morning story is reconstructable

Its own footprint is a sleeping process: RAM is the bottleneck here (15.5 GB, ~2 GB free with
training resident) and Python dies silently when it runs short.
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CONSOLE_LOG = Path("tests/pretrain_v2_console.log")
WATCHDOG_LOG = Path("tests/pretrain_v2_watchdog.log")
CHECKPOINT = Path("checkpoints/pretrain_v2_latest.pt")
RESULT_LOG = Path("tests/smoke_results/pretrain_v2_log.json")
PYTHON = Path(sys.executable)   # absolute, so a restart cannot depend on the cwd

POLL_SECONDS = 300
# The trainer prints every 200 steps, roughly every 165 s. 20 minutes of silence is well past
# any normal gap, including a validation pass plus a 911 MB checkpoint write.
SILENCE_LIMIT = 1200
MAX_RESTARTS = 3
BACKOFF_SECONDS = 120
TOTAL_TOKENS = 1_600_000_000
TOTAL_STEPS = 130_208


def note(message):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def trainer_running():
    """True if a python process is running training.pretrain_v2.

    Uses WMIC-style command line inspection through PowerShell rather than psutil, which is not
    installed and is not worth adding (requirements.txt would risk pulling CPU-only torch).
    """
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*pretrain_v2*' } | Measure-Object).Count"],
            capture_output=True, text=True, timeout=60)
        return int((result.stdout or "0").strip() or 0) > 0
    except Exception as error:                     # never let the check itself kill the watchdog
        note(f"process check failed ({error}) - assuming alive, will not restart")
        return True


def last_progress():
    if not CONSOLE_LOG.exists():
        return None
    return time.time() - CONSOLE_LOG.stat().st_mtime


def read_log_text():
    """Read the console log regardless of how PowerShell encoded it.

    Tee-Object writes UTF-16LE with a BOM on Windows PowerShell. Reading that as UTF-8 yields
    text interleaved with NULs, which survives .strip() and prints as nothing - so every watchdog
    report would look empty at exactly the moment it needed to be informative. Onboarding
    section 13 records the same trap with README.md being a UTF-16 stub.
    """
    raw = CONSOLE_LOG.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def tail(lines=3):
    """Last N NON-EMPTY lines.

    Taking the last N lines literally returns blanks, because Tee-Object leaves trailing empty
    lines - which would make every watchdog report say nothing at exactly the moment it matters.
    """
    if not CONSOLE_LOG.exists():
        return ""
    try:
        useful = [line.strip() for line in read_log_text().splitlines() if line.strip()]
        return " | ".join(useful[-lines:]) if useful else "(log is empty)"
    except Exception as error:
        return f"(could not read log: {error})"


def run_completed():
    """Did the pretrain FINISH, as opposed to dying?

    Without this the watchdog does the wrong thing at exactly the moment the run succeeds: the
    log goes quiet and the process exits, which looks identical to a crash, so it would try to
    restart a finished run three times. The trainer would refuse each time (start_step already
    exceeds max_steps), but the watchdog log would read like a failure when everything went
    right - and someone reading it in the morning would believe it.

    Two independent signals, either is enough:
      - the trainer's completion line in the console log
      - the result log reaching the final step
    """
    try:
        if CONSOLE_LOG.exists() and "done in" in read_log_text():
            return True, "trainer printed its completion line"
    except Exception:
        pass
    try:
        if RESULT_LOG.exists():
            data = json.loads(RESULT_LOG.read_text(encoding="utf-8"))
            history = data.get("history", [])
            if history and history[-1]["step"] >= TOTAL_STEPS - 1:
                return True, f"result log reached step {history[-1]['step']:,}"
    except Exception:
        pass
    return False, ""


def restart():
    if not CHECKPOINT.exists():
        note("NO CHECKPOINT - refusing to restart, a fresh run would silently discard nothing "
             "but would also mean the very first checkpoint was never written. Investigate.")
        return False
    note(f"restarting from {CHECKPOINT} ({CHECKPOINT.stat().st_size/1024**2:.0f} MB)")
    command = (f'$env:PYTHONIOENCODING="utf-8"; & "{PYTHON}" -u -m training.pretrain_v2 '
               f'--tokens {TOTAL_TOKENS} 2>&1 | Tee-Object -FilePath "{CONSOLE_LOG}" -Append')
    subprocess.Popen(["powershell", "-NoProfile", "-Command", command],
                     creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    return True


def main():
    restarts = 0
    note(f"watchdog started | poll {POLL_SECONDS}s | silence limit {SILENCE_LIMIT}s "
         f"| max restarts {MAX_RESTARTS}")
    while True:
        time.sleep(POLL_SECONDS)

        # Check completion on EVERY poll, not only after the silence threshold. Gating it behind
        # 20 minutes of quiet meant a successful finish took ~25 minutes to notice, which is 25
        # minutes of an idle GPU before the fine-tune can start.
        finished, why = run_completed()
        if finished:
            note(f"PRETRAIN COMPLETED ({why}). Exiting cleanly - nothing to restart.")
            note(f"  last output: {tail()}")
            note("  Next: ./.venv/Scripts/python.exe -m tools.finish_pretrain_and_finetune")
            return 0

        silence = last_progress()
        if silence is None:
            note("console log missing - nothing to watch yet")
            continue

        if silence < SILENCE_LIMIT:
            continue

        alive = trainer_running()
        note(f"log quiet for {silence/60:.1f} min | trainer alive: {alive}")
        note(f"  last output: {tail()}")

        if alive:
            note("  process is alive, so this is a slow stretch, not a death. Not intervening.")
            continue

        if restarts >= MAX_RESTARTS:
            note(f"  ALREADY RESTARTED {restarts} TIMES - stopping. Something is repeatably "
                 f"wrong and another restart would just burn the GPU. Needs a human.")
            return 1

        restarts += 1
        note(f"  trainer is gone. Restart attempt {restarts}/{MAX_RESTARTS}.")
        if not restart():
            return 1
        time.sleep(BACKOFF_SECONDS)
        note(f"  after restart, log activity: {tail(2)}")


if __name__ == "__main__":
    sys.exit(main())
