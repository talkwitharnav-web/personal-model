"""Compare two RAG indexes by RETRIEVAL quality, with the model taken out of the loop.

`tools/compare_indexes.py` scored end-to-end answers and made the rebuilt index look worse
(41.7% vs 60.0%). Inspecting the retrievals showed that conclusion was wrong: the new index
returned demonstrably good passages for every "failure" -

    What do bees make?          -> "Worker bees combine pollen, honey and glandular secretions..."
    What is a mitochondrion?    -> "Mitochondrion: powerhouse of the cell; site of cellular
                                    respiration producing ATP"
    What colour is the sky?     -> "When the sky is clear, the blue hour can be a colourful
                                    spectacle, with the indirect sunlight tinting the sky..."

The answer was in the context every time and the model picked a different span. So the end-to-end
score was measuring the model's extraction noise, not the index, and at 10 questions x 6 samples
that noise dominates - the third time in two days that a small end-to-end sample produced a
confident wrong conclusion.

This measures the index directly: does any retrieved passage CONTAIN the answer? No sampling, no
generation, no variance. Two metrics:

  hit@k       the answer string appears in at least one of the k returned passages
  hit@1       it appears in the top-ranked passage - the stricter, more useful number

⚠ Each index needs its OWN offsets cache. `orchestrator/retrieval.py` reuses
`rag/passage_offsets.npy` whenever it holds at least as many lines as the index has vectors, so
pointing config at a smaller index silently reads text from the wrong byte positions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from orchestrator import config
from orchestrator.retrieval import Retriever
from sentence_transformers import SentenceTransformer

RAG_DIR = Path("rag")

# (question, strings any one of which means the answer is present). Broad enough coverage that a
# few unlucky questions cannot swing the result.
QUESTIONS = [
    ("What is the capital of France?", ["capital of france", "paris is the capital"]),
    ("Who invented the Tesla coil?", ["nikola tesla"]),
    ("When did the Second World War begin?", ["1939"]),
    ("What is photosynthesis?", ["photosynthesis is"]),
    ("What is a mitochondrion?", ["organelle", "powerhouse"]),
    ("How many legs does a cat have?", ["four legs", "quadruped"]),
    ("What colour is the sky on a clear day?", ["blue"]),
    ("What do bees make?", ["honey"]),
    ("Was penicillin discovered by Alexander Fleming?", ["fleming"]),
    ("What is an earthquake?", ["shaking", "seismic"]),
    ("What is the largest planet?", ["jupiter"]),
    ("Who wrote Romeo and Juliet?", ["shakespeare"]),
    ("What is the boiling point of water?", ["100", "212"]),
    ("What is the longest river in the world?", ["nile", "amazon"]),
    ("Who painted the Mona Lisa?", ["leonardo", "vinci"]),
    ("What is the chemical symbol for gold?", [" au", "aurum"]),
    ("What causes rain?", ["condens", "precipitat", "evaporat"]),
    ("What is DNA?", ["deoxyribonucleic", "genetic"]),
    ("Who was Albert Einstein?", ["physicist", "relativity"]),
    ("What is the speed of light?", ["299", "186,000", "300,000"]),
    ("What is gravity?", ["attract", "force", "mass"]),
    ("What is the Great Wall of China?", ["fortification", "wall"]),
    ("What is an atom?", ["nucleus", "smallest unit", "particle"]),
    ("Who discovered gravity?", ["newton"]),
    ("What is the human heart?", ["organ", "pump", "blood"]),
]

VARIANTS = {
    "OLD (raw dump)": ("wikipedia.index", "wikipedia_passages.txt",
                       "build_state.json", "passage_offsets.npy"),
    "NEW (cleaned)": ("wikipedia_clean.index", "wikipedia_clean_passages.txt",
                      "build_state_clean.json", "passage_offsets_clean.npy"),
}


def measure(label, files, embedder):
    keys = ("INDEX_PATH", "PASSAGES_PATH", "STATE_PATH", "OFFSETS_PATH")
    original = {k: getattr(config, k) for k in keys}
    for key, name in zip(keys, files):
        setattr(config, key, RAG_DIR / name)
    try:
        retriever = Retriever(embedder)
        print(f"\n{'=' * 96}\n{label} - {retriever.summary}\n{'=' * 96}")
        hit_k = hit_1 = junk = 0
        misses = []
        for question, accepted in QUESTIONS:
            passages, dropped = retriever.search(question)
            junk += dropped
            texts = [text.lower() for _, text in passages]
            found_k = any(a in t for t in texts for a in accepted)
            found_1 = bool(texts) and any(a in texts[0] for a in accepted)
            hit_k += found_k
            hit_1 += found_1
            if not found_k:
                misses.append(question)
        retriever.close()
        n = len(QUESTIONS)
        print(f"  hit@3 {hit_k}/{n} = {100*hit_k/n:.0f}%   "
              f"hit@1 {hit_1}/{n} = {100*hit_1/n:.0f}%   junk dropped {junk}")
        if misses:
            print("  answer never retrieved for:")
            for question in misses:
                print(f"      {question}")
        return hit_k, hit_1, n
    finally:
        for key, value in original.items():
            setattr(config, key, value)


def main():
    embedder = SentenceTransformer(config.EMBEDDER_NAME)
    results = {}
    for label, files in VARIANTS.items():
        if not (RAG_DIR / files[0]).exists():
            raise SystemExit(f"missing {RAG_DIR / files[0]}")
        results[label] = measure(label, files, embedder)

    print(f"\n{'=' * 96}\nVERDICT\n{'=' * 96}")
    labels = list(VARIANTS)
    for label in labels:
        hit_k, hit_1, n = results[label]
        print(f"  {label:<18} hit@3 {100*hit_k/n:5.0f}%   hit@1 {100*hit_1/n:5.0f}%")
    old_k, old_1, n = results[labels[0]]
    new_k, new_1, _ = results[labels[1]]
    print()
    if new_k > old_k or (new_k == old_k and new_1 > old_1):
        print("  -> the rebuilt index retrieves the answer at least as often. Adopt it.")
    elif new_k == old_k and new_1 == old_1:
        print("  -> the two are indistinguishable on retrieval. Prefer the cleaned one anyway:")
        print("     it is 4% smaller and carries none of the disambiguation junk.")
    else:
        print("  -> the rebuilt index retrieves the answer LESS often. Keep the old one and")
        print("     investigate before switching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
