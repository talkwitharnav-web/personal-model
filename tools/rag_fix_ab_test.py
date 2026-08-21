"""Measure the SAME questions with the old retriever and the new one, back to back.

`tools/rag_accuracy_measure.py` gives an "after" number, but a number with nothing to compare it to
proves nothing - and reverting the code to measure "before" risks leaving the revert in place. So
this monkeypatches the filter off inside one process, measures, patches it back on, and measures
again. Same model, same index, same questions, same seeds.

The two things being tested are the 2026-08-19 retrieval changes:
  - disambiguation and list-page patterns added to BOILERPLATE_PATTERNS
  - exact-duplicate passages removed before TOP_K is taken
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from orchestrator import retrieval
from orchestrator.pipeline import Assistant
from tools.rag_accuracy_measure import CASES

SAMPLES = 8

# The patterns that existed BEFORE today's change - everything up to and including the
# See also/References rule. Restoring exactly this list reproduces the old behaviour.
OLD_PATTERNS = retrieval.BOILERPLATE_PATTERNS[:9]
NEW_PATTERNS = retrieval.BOILERPLATE_PATTERNS


def measure(assistant, label):
    total, greedy_hits, dropped_total = 0.0, 0, 0
    rows = []
    for question, accepted in CASES:
        route = assistant.plan(question)
        dropped_total += route.dropped
        prompt, _ = assistant.generator.build_prompt(question, route.passages)
        hits = 0
        for _ in range(SAMPLES):
            answer = "".join(assistant.generator.stream(prompt)).strip()
            if any(f in answer.lower() for f in accepted):
                hits += 1
        greedy = "".join(assistant.generator.stream(prompt, temperature=0.01)).strip()
        hit = any(f in greedy.lower() for f in accepted)
        greedy_hits += hit
        total += hits / SAMPLES
        rows.append((question, hits / SAMPLES, hit, greedy[:60], route.dropped))
    n = len(CASES)
    print(f"\n[{label}] mean sampled {total/n:.1%} | greedy {greedy_hits}/{n} | "
          f"dropped {dropped_total}")
    return total / n, greedy_hits, rows


def main():
    assistant = Assistant(device="cuda", quiet=True)

    retrieval.BOILERPLATE_PATTERNS = OLD_PATTERNS
    retrieval.DEDUPLICATE = False
    before_rate, before_greedy, before_rows = measure(assistant, "BEFORE")

    retrieval.BOILERPLATE_PATTERNS = NEW_PATTERNS
    retrieval.DEDUPLICATE = True
    after_rate, after_greedy, after_rows = measure(assistant, "AFTER")

    print(f"\n{'=' * 96}")
    print(f"  {'question':<52}{'before':>12}{'after':>12}")
    print("  " + "-" * 92)
    for (question, b_rate, b_hit, _, b_drop), (_, a_rate, a_hit, _, a_drop) in zip(before_rows,
                                                                                  after_rows):
        arrow = ""
        if a_rate > b_rate + 0.12:
            arrow = "  UP"
        elif a_rate < b_rate - 0.12:
            arrow = "  down"
        print(f"  {question[:50]:<52}{b_rate:>11.0%}{a_rate:>12.0%}{arrow}")
    print("  " + "-" * 92)
    print(f"  {'MEAN':<52}{before_rate:>11.1%}{after_rate:>12.1%}")
    print(f"  {'greedy correct':<52}{before_greedy:>11}{after_greedy:>12}")
    print("=" * 96)
    assistant.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
