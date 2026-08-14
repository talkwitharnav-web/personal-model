from tokenizers import Tokenizer
from pathlib import Path
import numpy as np

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
print(f"Loaded tokenizer with vocab size {vocab_size}")

def tokenize_file(input_path, output_path, chunk_size=20000):
    print(f"Reading {input_path}...")
    text = Path(input_path).read_text(encoding="utf-8")
    stories = text.split("<|endoftext|>")
    print(f"Encoding {len(stories)} documents in chunks of {chunk_size}...")

    endoftext_id = tokenizer.token_to_id("<|endoftext|>")
    total_tokens = 0
    with open(output_path, "wb") as out_file:
        for start in range(0, len(stories), chunk_size):
            chunk = stories[start:start + chunk_size]
            encodings = tokenizer.encode_batch(chunk)

            arrays = []
            for enc in encodings:
                arrays.append(np.array(enc.ids, dtype=np.uint16))
                arrays.append(np.array([endoftext_id], dtype=np.uint16))

            chunk_ids = np.concatenate(arrays)
            chunk_ids.tofile(out_file)
            total_tokens += len(chunk_ids)
            print(f"  {start + len(chunk):,}/{len(stories):,} documents -> {total_tokens:,} tokens so far")

    print(f"Saved {total_tokens:,} tokens to {output_path}")

output_dir = Path("data/dailydialog")

tokenize_file(output_dir / "train.txt", output_dir / "train.bin")
tokenize_file(output_dir / "validation.txt", output_dir / "validation.bin")
