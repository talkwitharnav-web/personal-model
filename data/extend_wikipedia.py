from datasets import load_dataset
from pathlib import Path

# Extends data/wikipedia/train.txt from the 1.1B-token training budget out to the
# whole corpus, for RAG coverage. Safe to append because the streamed shuffle uses
# a fixed seed, so articles come back in exactly the same order every run - verified
# by comparing the first articles of a fresh stream against the existing file.
train_path = Path("data/wikipedia/train.txt")
MARKER = b"<|endoftext|>"
READ_BYTES = 50_000_000
REPORT_EVERY = 5000

# Articles consumed from the stream when train.txt and validation.txt were first
# written (2,051,932 + 3,827), counted directly from those files. Everything after
# this point is new material.
SKIP_STREAM_ARTICLES = 2_055_759
# train.txt held this many articles before any appending started.
BASELINE_TRAIN_ARTICLES = 2_034_772


def scan(path):
    """Count articles and find the end of the last complete one, in bytes."""
    count = 0
    safe_end = 0
    position = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(READ_BYTES)
            if not chunk:
                break
            start = 0
            while True:
                found = chunk.find(MARKER, start)
                if found == -1:
                    break
                count += 1
                safe_end = position + found + len(MARKER)
                start = found + len(MARKER)
            position += len(chunk)
    return count, safe_end, position


print("Counting what is already on disk (this reads several GB, give it a minute)...")
articles, safe_end, size = scan(train_path)

# A run killed mid-write leaves a half-written article with no marker; cut it off
# so the file always ends on a clean boundary.
if size > safe_end:
    print(f"  trimming {size - safe_end:,} bytes of an incomplete trailing article")
    with open(train_path, "r+b") as f:
        f.truncate(safe_end)

appended_already = max(0, articles - BASELINE_TRAIN_ARTICLES)
skip = SKIP_STREAM_ARTICLES + appended_already
print(f"train.txt holds {articles:,} articles ({appended_already:,} of them appended so far)")
print(f"Skipping the first {skip:,} articles of the stream, appending everything after.")

dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
dataset = dataset.shuffle(seed=42, buffer_size=10000)

seen = 0
appended = 0
with open(train_path, "a", encoding="utf-8") as f:
    for example in dataset:
        text = example["text"].strip()
        if not text:
            continue  # must mirror prepare_wikipedia.py exactly or the skip count drifts
        seen += 1
        if seen <= skip:
            continue

        f.write(text + "\n" + MARKER.decode() + "\n")
        appended += 1

        if appended % REPORT_EVERY == 0:
            f.flush()
            print(f"  appended {appended:,} new articles ({articles + appended:,} total)")

print(f"Done. Appended {appended:,} articles; train.txt now holds {articles + appended:,}.")
print("Next: re-run `python -m rag.build_index` - it resumes from its saved state and "
      "indexes only the new tail.")
