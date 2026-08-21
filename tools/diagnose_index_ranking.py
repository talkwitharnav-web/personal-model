"""Why did the CLEANED index rank worse? Compare top-1 passages side by side.

Counter-intuitive result from `tools/compare_index_retrieval.py`:

    old (raw dump)   hit@3 80%   hit@1 68%
    new (cleaned)    hit@3 76%   hit@1 40%

Cleaning removed 608,225 junk articles and only 3.9% of passages, so the cleaned index SHOULD be
at least as good. hit@3 barely moved, which says the right passage is still in there - it is the
RANKING that got worse. This prints the top-ranked passage from each index for every question so
the difference can be read rather than guessed at.

Two candidate explanations worth separating:

  quantisation   `IndexIVFPQ` learns its cluster centroids and compression codebooks from the first
                 TRAINING_ARTICLES (50,000) of whatever file it is handed. A different corpus gives
                 different centroids, and a worse fit shows up as worse ranking rather than as
                 missing passages - which matches hit@3 holding while hit@1 collapses.

  content        cleaning strips markup in place and drops short articles, so passage boundaries
                 shift. If the cleaner removed the short, definition-shaped passages that used to
                 rank first, that would also do it.

The two look different in the output: quantisation noise gives a plausible-but-unrelated top hit,
while a content change gives a top hit that is clearly a different KIND of text.

Also reports McNemar's paired counts, because 17/25 versus 10/25 on the same questions is a paired
comparison and the raw percentages overstate how certain it is.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sentence_transformers import SentenceTransformer

from orchestrator import config
from orchestrator.retrieval import Retriever
from tools.compare_index_retrieval import QUESTIONS, VARIANTS, RAG_DIR


def top_hits(files, embedder):
    keys = ("INDEX_PATH", "PASSAGES_PATH", "STATE_PATH", "OFFSETS_PATH")
    original = {k: getattr(config, k) for k in keys}
    for key, name in zip(keys, files):
        setattr(config, key, RAG_DIR / name)
    try:
        retriever = Retriever(embedder)
        out = {}
        for question, accepted in QUESTIONS:
            passages, _ = retriever.search(question)
            top = passages[0] if passages else (0.0, "")
            found_1 = any(a in top[1].lower() for a in accepted)
            out[question] = (found_1, top[0], top[1])
        retriever.close()
        return out
    finally:
        for key, value in original.items():
            setattr(config, key, value)


def main():
    embedder = SentenceTransformer(config.EMBEDDER_NAME)
    labels = list(VARIANTS)
    old = top_hits(VARIANTS[labels[0]], embedder)
    new = top_hits(VARIANTS[labels[1]], embedder)

    old_only = new_only = both = neither = 0
    print(f"{'=' * 100}\nTOP-1 PASSAGE, SIDE BY SIDE (only where the two disagree)\n{'=' * 100}")
    for question, _ in QUESTIONS:
        o_hit, o_score, o_text = old[question]
        n_hit, n_score, n_text = new[question]
        both += o_hit and n_hit
        neither += not o_hit and not n_hit
        old_only += o_hit and not n_hit
        new_only += n_hit and not o_hit
        if o_hit == n_hit:
            continue
        winner = "OLD wins" if o_hit else "NEW wins"
        print(f"\n  {question}   [{winner}]")
        print(f"     old {o_score:.3f} {'HIT ' if o_hit else 'miss'}  {o_text[:88]}")
        print(f"     new {n_score:.3f} {'HIT ' if n_hit else 'miss'}  {n_text[:88]}")

    print(f"\n{'=' * 100}\nPAIRED COUNTS (McNemar)\n{'=' * 100}")
    print(f"  both indexes hit@1      {both}")
    print(f"  only OLD hit@1          {old_only}")
    print(f"  only NEW hit@1          {new_only}")
    print(f"  neither                 {neither}")
    discordant = old_only + new_only
    print(f"\n  discordant pairs: {discordant} ({old_only} favour old, {new_only} favour new)")
    if discordant:
        # Exact binomial two-sided p under H0: a discordant pair favours either index equally.
        from math import comb
        k = min(old_only, new_only)
        p = 2 * sum(comb(discordant, i) for i in range(k + 1)) / (2 ** discordant)
        p = min(p, 1.0)
        print(f"  exact two-sided p = {p:.4f}")
        if p < 0.05:
            print("  -> the difference is unlikely to be chance. Keep the old index.")
        else:
            print("  -> NOT statistically significant at this sample size. The direction favours")
            print("     the old index but more questions would be needed to be sure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
