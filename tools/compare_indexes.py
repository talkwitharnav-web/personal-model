"""Compare the old RAG index against the rebuilt one, before anything is switched over.

The original index was built 2026-08-15 from the RAW Wikipedia dump - two days before
`data/clean_wikipedia.py` existed - so every disambiguation page, list stub and template wreck went
straight into retrieval. The rebuild (2026-08-20) uses the fully cleaned corpus:

    old   61,063,922 passages from 6,386,829 articles   (raw dump)
    new   58,714,131 passages from 5,778,604 articles   (cleaned, 96.1% of passages kept)

The coverage loss is small and is exactly the junk: disambiguation pages, list stubs and articles
under 200 characters.

⚠ THE TRAP THIS FILE EXISTS TO AVOID. `orchestrator/retrieval.py` caches passage byte offsets in
ONE file, `rag/passage_offsets.npy`, and reuses the cache whenever it holds at least as many lines
as the index has vectors. The old cache has 61.0M entries and the new index has 58.7M vectors, so
naively pointing config at the new passages file would silently reuse the OLD offsets - every
lookup would return text from the wrong byte position, and the answers would look like nonsense
for a reason having nothing to do with index quality. Each index therefore gets its own offsets
file here.

Nothing is switched over by this script. It prints the numbers and stops.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from orchestrator import config
from orchestrator.pipeline import Assistant
from tools.rag_accuracy_measure import CASES

SAMPLES = 6
RAG_DIR = Path("rag")

VARIANTS = {
    "OLD (raw dump)": {
        "INDEX_PATH": RAG_DIR / "wikipedia.index",
        "PASSAGES_PATH": RAG_DIR / "wikipedia_passages.txt",
        "STATE_PATH": RAG_DIR / "build_state.json",
        "OFFSETS_PATH": RAG_DIR / "passage_offsets.npy",
    },
    "NEW (cleaned)": {
        "INDEX_PATH": RAG_DIR / "wikipedia_clean.index",
        "PASSAGES_PATH": RAG_DIR / "wikipedia_clean_passages.txt",
        "STATE_PATH": RAG_DIR / "build_state_clean.json",
        # Its own cache - see the docstring. Reusing the old one returns text from wrong offsets.
        "OFFSETS_PATH": RAG_DIR / "passage_offsets_clean.npy",
    },
}


def measure(label, paths):
    original = {key: getattr(config, key) for key in paths}
    for key, value in paths.items():
        setattr(config, key, value)
    try:
        assistant = Assistant(device="cuda", quiet=True)
        print(f"\n{'=' * 96}\n{label} - {assistant.retriever.summary}\n{'=' * 96}")
        total, greedy_hits, dropped = 0.0, 0, 0
        rows = []
        for question, accepted in CASES:
            route = assistant.plan(question)
            dropped += route.dropped
            prompt, _ = assistant.generator.build_prompt(question, route.passages)
            hits = 0
            for _ in range(SAMPLES):
                answer = "".join(assistant.generator.stream(prompt)).strip()
                if any(a in answer.lower() for a in accepted):
                    hits += 1
            greedy = "".join(assistant.generator.stream(prompt, temperature=0.01)).strip()
            hit = any(a in greedy.lower() for a in accepted)
            greedy_hits += hit
            total += hits / SAMPLES
            rows.append((question, hits / SAMPLES))
            top = route.passages[0][1][:58] if route.passages else "(none)"
            print(f"  {hits}/{SAMPLES}  {'HIT ' if hit else 'miss'}  {question[:44]:<46} {greedy[:34]}")
            print(f"           top passage: {top}")
        assistant.close()
        n = len(CASES)
        print(f"\n  mean sampled {total/n:.1%} | greedy {greedy_hits}/{n} | junk dropped {dropped}")
        return total / n, greedy_hits, rows
    finally:
        for key, value in original.items():
            setattr(config, key, value)


def main():
    for label, paths in VARIANTS.items():
        if not paths["INDEX_PATH"].exists():
            raise SystemExit(f"missing {paths['INDEX_PATH']}")

    results = {}
    for label, paths in VARIANTS.items():
        results[label] = measure(label, paths)

    print(f"\n{'=' * 96}\nSIDE BY SIDE\n{'=' * 96}")
    labels = list(VARIANTS)
    old_rows = dict((q, r) for q, r in results[labels[0]][2])
    new_rows = dict((q, r) for q, r in results[labels[1]][2])
    print(f"  {'question':<50}{'old':>10}{'new':>10}")
    print("  " + "-" * 70)
    for question, _ in CASES:
        o, n = old_rows[question], new_rows[question]
        arrow = "  UP" if n > o + 0.15 else ("  down" if n < o - 0.15 else "")
        print(f"  {question[:48]:<50}{o:>9.0%}{n:>10.0%}{arrow}")
    print("  " + "-" * 70)
    print(f"  {'MEAN':<50}{results[labels[0]][0]:>9.1%}{results[labels[1]][0]:>10.1%}")
    print(f"  {'greedy correct':<50}{results[labels[0]][1]:>9}{results[labels[1]][1]:>10}")
    print("\nNothing has been switched. To adopt the new index, point INDEX_PATH, PASSAGES_PATH,")
    print("STATE_PATH and OFFSETS_PATH in orchestrator/config.py at the *_clean files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
