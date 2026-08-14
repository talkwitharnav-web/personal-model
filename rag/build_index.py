from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = model.get_sentence_embedding_dimension()

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

def article_to_passages(article):
    passages = []
    for paragraph in article.split("\n"):
        paragraph = paragraph.strip()
        if len(paragraph) > 50:
            passages.append(paragraph)
    return passages

output_dir = Path("rag")
output_dir.mkdir(parents=True, exist_ok=True)
passages_path = output_dir / "wikipedia_passages.txt"

index = faiss.IndexFlatIP(embedding_dim)

ARTICLES_PER_BATCH = 5000
# IndexFlatIP keeps every vector permanently in RAM - it crashed with MemoryError
# at ~1,185,000 articles (~9.68M passages, ~14.9GB just for the vectors). Capping
# well under that with real safety margin, since Wikipedia training also shares RAM.
MAX_ARTICLES = 600_000
batch_articles = []
article_count = 0
total_passages = 0

def embed_and_index_batch(batch_articles, passages_file):
    batch_passages = []
    for article in batch_articles:
        batch_passages.extend(article_to_passages(article))

    if not batch_passages:
        return 0

    embeddings = model.encode(batch_passages, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    for passage in batch_passages:
        passages_file.write(passage.replace("\n", " ") + "\n")

    return len(batch_passages)

with open(passages_path, "w", encoding="utf-8") as passages_file:
    for article in iter_articles("data/wikipedia/train.txt"):
        if article_count >= MAX_ARTICLES:
            break

        batch_articles.append(article)
        article_count += 1

        if len(batch_articles) >= ARTICLES_PER_BATCH:
            total_passages += embed_and_index_batch(batch_articles, passages_file)
            batch_articles = []
            print(f"{article_count:,} articles processed -> {total_passages:,} passages indexed so far")

    # handle any leftover articles that didn't fill a final full batch
    if batch_articles:
        total_passages += embed_and_index_batch(batch_articles, passages_file)
        print(f"{article_count:,} articles processed -> {total_passages:,} passages indexed so far")

faiss.write_index(index, str(output_dir / "wikipedia.index"))
print(f"Done. Indexed {total_passages:,} passages from {article_count:,} articles.")
print(f"Saved index to {output_dir / 'wikipedia.index'}, passages to {passages_path}")
