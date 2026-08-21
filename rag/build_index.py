from pathlib import Path
from sentence_transformers import SentenceTransformer
import argparse
import faiss
import numpy as np
import json
import os
import torch

# Rebuilt 2026-08-19 from the CLEANED corpus. The original index was built on 2026-08-15 from the
# raw dump, two days before data/clean_wikipedia.py existed, so it carried every disambiguation
# page, list stub and template wreck straight into retrieval. Measured cost: "What is the capital
# of France?" retrieved 'Paris is the capital of France, which may consist of :' followed by
# 'It may also refer to the following places in France:' TWICE, and the model answered "Founded in
# the 19th century".
#
# ⚠ Pass --target-gb 0 to data/clean_wikipedia.py first. Its default caps output at 7.5 GB - that
# is the PRETRAIN's byte budget, not a limit of the cleaner, and cleaning to the default would
# silently index only 40% of Wikipedia.
parser = argparse.ArgumentParser()
parser.add_argument("--source", default="data/pretrain/wikipedia_clean.txt")
parser.add_argument("--suffix", default="",
                    help="output suffix, e.g. '_clean' writes rag/wikipedia_clean.index so the "
                         "live index keeps serving until the new one is verified")
parser.add_argument("--max-articles", type=int, default=None)
args = parser.parse_args()

model = SentenceTransformer("all-MiniLM-L6-v2",
                            device="cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = model.get_embedding_dimension()
print(f"embedding on {model.device}")

def iter_articles(path, read_bytes=50_000_000):
    # Same streaming pattern as tokenize_wikipedia.py's iter_documents -
    # reading the whole 5.69GB file at once causes a MemoryError.
    buffer = ""
    with open(path, "r", encoding="utf-8") as f:
        while True:
            data = f.read(read_bytes)
            if not data:
                break
            buffer += data
            parts = buffer.split("<|endoftext|>")
            buffer = parts.pop()
            for article in parts:
                yield article
    if buffer:
        yield buffer

MIN_PASSAGE_CHARS = 50
# MiniLM silently truncates anything past max_seq_length (256 word-pieces, ~1000
# chars) - split long paragraphs into overlapping chunks instead of losing the tail.
MAX_CHARS_PER_PASSAGE = 1000
PASSAGE_OVERLAP = 100

def article_to_passages(article):
    passages = []
    for paragraph in article.split("\n"):
        paragraph = paragraph.strip()
        if len(paragraph) <= MIN_PASSAGE_CHARS:
            continue
        if len(paragraph) <= MAX_CHARS_PER_PASSAGE:
            passages.append(paragraph)
            continue
        step = MAX_CHARS_PER_PASSAGE - PASSAGE_OVERLAP
        for start in range(0, len(paragraph), step):
            chunk = paragraph[start:start + MAX_CHARS_PER_PASSAGE].strip()
            if len(chunk) > MIN_PASSAGE_CHARS:
                passages.append(chunk)
    return passages

WIKIPEDIA_PATH = args.source

output_dir = Path("rag")
output_dir.mkdir(parents=True, exist_ok=True)
passages_path = output_dir / f"wikipedia{args.suffix}_passages.txt"
index_path = output_dir / f"wikipedia{args.suffix}.index"
state_path = output_dir / f"build_state{args.suffix}.json"

ARTICLES_PER_BATCH = 5000
BATCHES_PER_CHECKPOINT = 4
MAX_ARTICLES = args.max_articles  # None = index the entire corpus

# IndexIVFPQ (product quantization) stores each 384-dim vector in PQ_SUBVECTORS
# bytes instead of 1,536 - a 32x reduction, so the whole ~16.5M-passage corpus
# fits in well under 1GB. The old IndexFlatIP needed ~7GB for a quarter of that
# and died with MemoryError at ~9.68M vectors.
NLIST = 4096                # coarse clusters; rule of thumb is ~sqrt(total vectors)
PQ_SUBVECTORS = 48          # bytes per stored vector; must divide embedding_dim evenly
PQ_BITS = 8
SEARCH_NPROBE = 16          # clusters scanned per query - higher is more accurate, slower
TRAINING_ARTICLES = 50_000  # yields ~400k sample vectors, ~100x NLIST per FAISS guidance

if embedding_dim % PQ_SUBVECTORS != 0:
    raise SystemExit(
        f"PQ_SUBVECTORS ({PQ_SUBVECTORS}) must divide embedding_dim ({embedding_dim}) evenly"
    )

def load_state():
    if not (index_path.exists() and passages_path.exists() and state_path.exists()):
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

def build_trained_index():
    # Unlike IndexFlatIP, IVFPQ must learn its cluster centroids and compression
    # codebooks from a representative sample before anything can be added.
    # train.txt was already randomly shuffled when downloaded, so the first N
    # articles are effectively a random sample of the corpus.
    print(f"Collecting a training sample from the first {TRAINING_ARTICLES:,} articles...")
    sample_passages = []
    for i, article in enumerate(iter_articles(WIKIPEDIA_PATH)):
        if i >= TRAINING_ARTICLES:
            break
        sample_passages.extend(article_to_passages(article))

    print(f"Embedding {len(sample_passages):,} training passages...")
    sample = model.encode(sample_passages, convert_to_numpy=True, show_progress_bar=True)
    sample = sample.astype("float32")
    faiss.normalize_L2(sample)

    quantizer = faiss.IndexFlatIP(embedding_dim)
    new_index = faiss.IndexIVFPQ(
        quantizer, embedding_dim, NLIST, PQ_SUBVECTORS, PQ_BITS, faiss.METRIC_INNER_PRODUCT
    )
    # Hand the quantizer's lifetime to the index so Python's GC can't free it early.
    new_index.own_fields = True
    quantizer.this.disown()

    print(f"Training index on {sample.shape[0]:,} vectors (nlist={NLIST}, {PQ_SUBVECTORS} bytes/vector)...")
    new_index.train(sample)
    new_index.nprobe = SEARCH_NPROBE
    print("Training complete")
    return new_index

state = load_state()
if state is None:
    index = build_trained_index()
    articles_done = 0
    safe_byte_offset = 0
    passages_file = open(passages_path, "w", encoding="utf-8")
    print("Starting a fresh index build")
else:
    index = faiss.read_index(str(index_path))
    if not isinstance(index, faiss.IndexIVFPQ):
        raise SystemExit(
            f"{index_path} is not an IndexIVFPQ - it is probably an older uncompressed "
            f"build. Delete rag/build_state.json, rag/wikipedia.index and "
            f"rag/wikipedia_passages.txt to rebuild from scratch."
        )
    index.nprobe = SEARCH_NPROBE
    if index.ntotal != state["passages_count"]:
        raise SystemExit(
            f"Corrupt state: index holds {index.ntotal:,} vectors but state claims "
            f"{state['passages_count']:,}. Delete rag/build_state.json, "
            f"rag/wikipedia.index and rag/wikipedia_passages.txt to rebuild from scratch."
        )
    articles_done = state["article_count"]
    safe_byte_offset = state["passages_byte_offset"]
    # Passages are written before the index is updated, so the file can be AHEAD
    # of the saved index - cut it back so line N always matches vector N.
    with open(passages_path, "r+b") as f:
        f.truncate(safe_byte_offset)
    passages_file = open(passages_path, "a", encoding="utf-8")
    print(f"Resuming: {articles_done:,} articles / {index.ntotal:,} passages already indexed")

def embed_and_index_batch(batch_articles):
    global safe_byte_offset

    batch_passages = []
    for article in batch_articles:
        batch_passages.extend(article_to_passages(article))
    if not batch_passages:
        return

    # Write passages first: a file that runs ahead of the index is recoverable
    # (truncated on resume), an index that runs ahead is not - its vectors would
    # have no text to look up.
    for passage in batch_passages:
        passages_file.write(passage + "\n")
    passages_file.flush()

    embeddings = model.encode(batch_passages, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Only now are these passages safely represented in the index.
    safe_byte_offset = os.path.getsize(passages_path)

def save_checkpoint(articles_done):
    passages_file.flush()
    os.fsync(passages_file.fileno())
    faiss.write_index(index, str(index_path))
    state_path.write_text(json.dumps({
        "article_count": articles_done,
        "passages_count": index.ntotal,
        "passages_byte_offset": safe_byte_offset,
    }), encoding="utf-8")
    print(f"  checkpoint saved: {articles_done:,} articles / {index.ntotal:,} passages")

batch_articles = []
article_count = 0
batches_since_checkpoint = 0

try:
    for article in iter_articles(WIKIPEDIA_PATH):
        if MAX_ARTICLES is not None and article_count >= MAX_ARTICLES:
            break
        article_count += 1
        if article_count <= articles_done:
            continue

        batch_articles.append(article)
        if len(batch_articles) < ARTICLES_PER_BATCH:
            continue

        embed_and_index_batch(batch_articles)
        batch_articles = []
        print(f"{article_count:,} articles processed -> {index.ntotal:,} passages indexed so far")

        batches_since_checkpoint += 1
        if batches_since_checkpoint >= BATCHES_PER_CHECKPOINT:
            save_checkpoint(article_count)
            batches_since_checkpoint = 0

    if batch_articles:
        embed_and_index_batch(batch_articles)
        print(f"{article_count:,} articles processed -> {index.ntotal:,} passages indexed so far")

    save_checkpoint(article_count)
except KeyboardInterrupt:
    print("\nInterrupted - saving progress so the next run can resume...")
    save_checkpoint(article_count - len(batch_articles))
    raise
finally:
    passages_file.close()

print(f"Done. Indexed {index.ntotal:,} passages from {article_count:,} articles.")
print(f"Index: {index_path}  Passages: {passages_path}")
