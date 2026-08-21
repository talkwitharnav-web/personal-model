"""Wait for the Wikipedia clean to finish, then rebuild the RAG index from the cleaned corpus.

Chained rather than run by hand because the clean takes hours and the rebuild takes hours more,
and nobody should have to sit between them. Dark asked for the rebuild on 2026-08-19 after the
original index was found to have been built on 2026-08-15 from the RAW dump - two days before
`data/clean_wikipedia.py` existed - which is why "What is the capital of France?" retrieved
'Paris is the capital of France, which may consist of :' followed by the same boilerplate line
twice, and the model answered "Founded in the 19th century".

⚠ The clean MUST have been started with `--target-gb 0`. Its default caps output at 7.5 GB, which
is the PRETRAIN's byte budget and not a limit of the cleaner; rebuilding from a capped clean would
index roughly 40% of Wikipedia and silently lose most of the knowledge base.

The new index is written to `rag/wikipedia_clean.*` so the live one keeps serving throughout. It
is NOT switched over automatically - that is a decision for a human, after comparing the two with
tools/rag_accuracy_measure.py.
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
CLEAN_LOG = ROOT / "logs" / "clean_full.log"
CLEANED = ROOT / "data" / "pretrain" / "wikipedia_clean.txt"
BUILD_LOG = ROOT / "logs" / "index_rebuild.log"
NOTE_LOG = ROOT / "tests" / "index_rebuild_chain.log"

POLL_SECONDS = 120
# The clean prints a summary block when it finishes. Waiting for the file to stop growing is not
# enough on its own - it stalls for minutes at a time on large articles.
DONE_MARKERS = ("wikipedia_clean_validation.txt", "% retained")


def note(message):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    NOTE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTE_LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def clean_running():
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*clean_wikipedia*' } | Measure-Object).Count"],
            capture_output=True, text=True, timeout=120)
        return int((result.stdout or "0").strip() or 0) > 0
    except Exception as error:
        note(f"process check failed ({error}) - assuming still running")
        return True


def main():
    note("waiting for the Wikipedia clean to finish before rebuilding the index")
    while True:
        if not clean_running():
            break
        time.sleep(POLL_SECONDS)

    size_gb = CLEANED.stat().st_size / 1024 ** 3 if CLEANED.exists() else 0
    note(f"clean finished - {CLEANED.name} is {size_gb:.2f} GB")

    # A capped clean would quietly halve the knowledge base, so refuse rather than build one.
    if size_gb < 12:
        note(f"REFUSING to rebuild: {size_gb:.2f} GB is far below the ~17 GB a full clean of the "
             f"18.46 GB dump should produce. Was --target-gb 0 passed? Nothing has been changed.")
        return 1

    note("starting the index rebuild into rag/wikipedia_clean.* (the live index keeps serving)")
    BUILD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOG, "w", encoding="utf-8") as out:
        process = subprocess.Popen(
            [str(PYTHON), "-u", "rag/build_index.py",
             "--source", "data/pretrain/wikipedia_clean.txt", "--suffix", "_clean"],
            cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)
        note(f"build started, PID {process.pid} - progress in {BUILD_LOG}")
        code = process.wait()

    note(f"index build exited with code {code}")
    if code == 0:
        note("NEXT: compare old against new before switching. The new index is NOT live yet.")
        note("  1. point orchestrator/config.py INDEX_PATH/PASSAGES_PATH/STATE_PATH at the")
        note("     *_clean files, or rename them over the originals (keep a backup)")
        note("  2. python -m tools.rag_accuracy_measure --tag NEW")
        note("     and compare with the 61.3% mean hit rate the current index scores")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
