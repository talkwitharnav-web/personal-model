from tokenizers import Tokenizer
from pathlib import Path
from model.config import DarkGLMConfig
import numpy as np

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
config = DarkGLMConfig()
pad_id = tokenizer.token_to_id("<|endoftext|>")
marker = tokenizer.encode("\nAssistant:").ids[1:]

def tokenize_example(text):
    full_ids = tokenizer.encode(text).ids
    # XSum articles are often long - many will exceed context_length and get
    # skipped below. Reported honestly via the skip counter, not fixed here.
    if len(full_ids) > config.context_length:
        return None

    answer_start = None
    for i in range(len(full_ids) - len(marker) + 1):
        if full_ids[i:i + len(marker)] == marker:
            answer_start = i + len(marker)
    if answer_start is None:
        return None

    labels = list(full_ids)
    for i in range(answer_start):
        labels[i] = -100

    pad_amount = config.context_length - len(full_ids)
    full_ids = full_ids + [pad_id] * pad_amount
    labels = labels + [-100] * pad_amount
    return full_ids, labels

def tokenize_file(input_path, ids_out_path, labels_out_path):
    print(f"Reading {input_path}...")
    text = Path(input_path).read_text(encoding="utf-8")
    documents = [doc for doc in text.split("<|endoftext|>") if doc.strip()]

    all_ids, all_labels = [], []
    skipped = 0
    for doc in documents:
        result = tokenize_example(doc)
        if result is None:
            skipped += 1
            continue
        ids, labels = result
        all_ids.append(ids)
        all_labels.append(labels)

    ids_array = np.array(all_ids, dtype=np.int32)
    labels_array = np.array(all_labels, dtype=np.int32)
    np.save(ids_out_path, ids_array)
    np.save(labels_out_path, labels_array)
    print(f"Saved {len(all_ids)} examples ({skipped} skipped as too long) to {ids_out_path}")

output_dir = Path("data/xsum")
tokenize_file(output_dir / "train.txt", output_dir / "train_ids.npy", output_dir / "train_labels.npy")
tokenize_file(output_dir / "validation.txt", output_dir / "validation_ids.npy", output_dir / "validation_labels.npy")
