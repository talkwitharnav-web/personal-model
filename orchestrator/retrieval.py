import json
import re

import faiss
import numpy as np

from orchestrator import config

# Wikipedia is full of citations, bibliographies and see-also lines. They match
# queries well because they carry all the right words but contain no facts, and
# they were crowding out real content ("temperature in the Arctic" returned three
# of them and nothing else).
BOILERPLATE_PATTERNS = (
    re.compile(r"https?://|www\."),
    re.compile(r"\bISBN\b|\bdoi:|\bOCLC\b|\bJSTOR\b|\barXiv\b"),
    re.compile(r"\bpp\.\s*\d|\bp\.\s*\d+\b"),
    re.compile(r"Archived from the original|Retrieved \d|Retrieved on|linked here"),
    re.compile(r"\b(?:Press|Publishers?|Publishing)\b\s*,"),
    re.compile(r"\bBibliography\b"),
    re.compile(r"^[A-Z][A-Za-z'\-]+, [A-Z][a-z]*\.?[\s,]"),   # "Serreze, Mark C. and ..."
    re.compile(r"\(\d{4}\)\.?\s*$"),
    re.compile(r"^\s*(?:See also|References|External links|Further reading|Notes|Sources)\b", re.I),
    # Disambiguation and navigation stubs. These are the single worst offenders for short
    # factual questions, because a disambiguation page is BUILT to match a bare topic name:
    # "What is the capital of France?" retrieved 'Paris is the capital of France, which may
    # consist of :' at 0.726 followed by 'It may also refer to the following places in France:'
    # TWICE, and the model answered "Founded in the 19th century". The real article ranked
    # below all three. Dropping these lets the next candidate take the slot.
    re.compile(r"\bmay (?:also )?refer to\b", re.I),
    re.compile(r"\bwhich may consist of\b", re.I),
    re.compile(r"\bdisambiguation\b", re.I),
    re.compile(r"^\s*(?:List|Index) of\b", re.I),
    re.compile(r"\bis a (?:surname|given name|family name)\b", re.I),
)


def is_boilerplate(passage):
    if any(pattern.search(passage) for pattern in BOILERPLATE_PATTERNS):
        return True
    # Glossary / see-also entries: "Topic – short description", no sentence ending.
    return " – " in passage and not passage.rstrip().endswith(".")


# Module-level so tools/rag_fix_ab_test.py can switch it off and measure the same questions
# with and without, in one process. Turning it off is only for measurement.
DEDUPLICATE = True


def _build_offsets(line_count):
    # Vector N in the index is line N of the passages file, so one byte offset per
    # line is enough to look any passage up without holding the 16GB file in memory.
    offsets = np.zeros(line_count, dtype=np.int64)
    written = 0
    offset = 0
    with open(config.PASSAGES_PATH, "rb") as f:
        for line in f:
            if written >= line_count:
                break
            offsets[written] = offset
            written += 1
            offset += len(line)
    if written < line_count:
        raise SystemExit(
            f"{config.PASSAGES_PATH} has {written:,} lines but the index holds "
            f"{line_count:,} vectors - they are out of sync."
        )
    return offsets


def _load_offsets(line_count):
    if config.OFFSETS_PATH.exists():
        cached = np.load(config.OFFSETS_PATH, mmap_mode="r")
        if len(cached) >= line_count:
            return cached
        # Windows refuses to overwrite a memory-mapped file, and this cache goes
        # stale every time the index grows.
        cached._mmap.close()
        del cached

    print("Mapping passage line offsets (one-off scan, cached afterwards)...")
    offsets = _build_offsets(line_count)
    np.save(config.OFFSETS_PATH, offsets)
    return offsets


class Retriever:
    """Finds real Wikipedia passages for a question."""

    def __init__(self, embedder):
        for path in (config.INDEX_PATH, config.PASSAGES_PATH, config.STATE_PATH):
            if not path.exists():
                raise SystemExit(f"Missing {path} - run `python -m rag.build_index` first.")

        self.embedder = embedder
        self.index = faiss.read_index(str(config.INDEX_PATH))
        self.index.nprobe = config.SEARCH_NPROBE
        self.offsets = _load_offsets(self.index.ntotal)
        self.passages_file = open(config.PASSAGES_PATH, "rb")
        self.state = json.loads(config.STATE_PATH.read_text(encoding="utf-8"))

    @property
    def summary(self):
        return (f"{self.index.ntotal:,} passages from "
                f"{self.state['article_count']:,} articles")

    def _read(self, position):
        self.passages_file.seek(int(self.offsets[position]))
        return self.passages_file.readline().decode("utf-8").rstrip("\n")

    def search(self, question):
        vector = self.embedder.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(vector)

        scores, positions = self.index.search(vector, config.CANDIDATE_K)
        candidates = [
            (float(score), self._read(position))
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        ]

        kept = [c for c in candidates if not is_boilerplate(c[1])]
        dropped = len(candidates) - len(kept)

        # De-duplicate. The index holds the same boilerplate line many times over (measured:
        # 2.4% of passages are exact duplicates, one repeated 321 times), and a duplicate
        # wastes one of only TOP_K=3 slots - a query that pulls the same line twice has just
        # one real passage left to answer from. Seen once already: 'It may also refer to the
        # following places in France:' occupied two of the three slots for one question.
        unique, seen = [], set()
        for score, text in kept:
            key = " ".join(text.split()).lower()
            if DEDUPLICATE and key in seen:
                dropped += 1
                continue
            seen.add(key)
            unique.append((score, text))

        # Never return nothing - if it was all boilerplate, these beat no context.
        return (unique or candidates)[:config.TOP_K], dropped

    def close(self):
        self.passages_file.close()
