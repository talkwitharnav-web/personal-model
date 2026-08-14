from tokenizers import Tokenizer
from pathlib import Path
import numpy as np

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
print(f"Loaded tokenizer with vocab size {vocab_size}")

endoftext_id = tokenizer.token_to_id("<|endoftext|>")

def iter_documents(path, read_bytes=50_000_000):
    # Stream the file in chunks instead of Path.read_text() - a multi-GB file
    # decoded as one Python string blows past available RAM (hit this on Wikipedia).
    buffer = ""
    with open(path, "r", encoding="utf-8") as f:
        while True:
            data = f.read(read_bytes)
            if not data:
                break
            buffer += data
            parts = buffer.split("<|endoftext|>")
            buffer = parts.pop()
            for doc in parts:
                yield doc
    if buffer:
        yield buffer

def tokenize_file(input_path, output_path, token_target, chunk_size=20000):
    print(f"Streaming {input_path} (target {token_target:,} tokens)...")
    total_tokens = 0
    total_docs = 0
    batch = []

    with open(output_path, "wb") as out_file:
        for doc in iter_documents(input_path):
            if total_tokens >= token_target:
                break
            batch.append(doc)
            if len(batch) < chunk_size:
                continue

            encodings = tokenizer.encode_batch(batch)
            arrays = []
            for enc in encodings:
                arrays.append(np.array(enc.ids, dtype=np.uint16))
                arrays.append(np.array([endoftext_id], dtype=np.uint16))
            chunk_ids = np.concatenate(arrays)

            remaining = token_target - total_tokens
            if len(chunk_ids) > remaining:
                chunk_ids = chunk_ids[:remaining]

            chunk_ids.tofile(out_file)
            total_tokens += len(chunk_ids)
            total_docs += len(batch)
            batch = []
            print(f"  {total_docs:,} documents -> {total_tokens:,} tokens so far")

        if batch and total_tokens < token_target:
            encodings = tokenizer.encode_batch(batch)
            arrays = []
            for enc in encodings:
                arrays.append(np.array(enc.ids, dtype=np.uint16))
                arrays.append(np.array([endoftext_id], dtype=np.uint16))
            chunk_ids = np.concatenate(arrays)

            remaining = token_target - total_tokens
            if len(chunk_ids) > remaining:
                chunk_ids = chunk_ids[:remaining]

            chunk_ids.tofile(out_file)
            total_tokens += len(chunk_ids)
            total_docs += len(batch)
            print(f"  {total_docs:,} documents -> {total_tokens:,} tokens so far")

    print(f"Saved {total_tokens:,} tokens to {output_path}")
    return total_tokens

output_dir = Path("data/wikipedia")

TRAIN_TOKEN_TARGET = 1_100_000_000
VALID_TOKEN_TARGET = 3_000_000

train_tokens = tokenize_file(output_dir / "train.txt", output_dir / "train.bin", TRAIN_TOKEN_TARGET)
valid_tokens = tokenize_file(output_dir / "validation.txt", output_dir / "validation.bin", VALID_TOKEN_TARGET)

print(f"Final: train={train_tokens:,} tokens, validation={valid_tokens:,} tokens")
