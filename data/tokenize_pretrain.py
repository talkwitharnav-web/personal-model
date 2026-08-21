"""Step 4c: tokenize the mixed pretraining corpus with the REFITTED tokenizer.

Uses tokenizer/tokenizer_v2.json (fitted on the real corpus, +9.42% compression over v1), so the
output .bin files are NOT interchangeable with data/wikipedia/*.bin, which were produced with the
old tokenizer. New paths, and the old files are left alone: a token stream and the tokenizer that
produced it are a matched pair, and the LR-sweep results are tied to the old ones.

MEMORY, which is the real risk here:
  tokenizer.encode_batch returns Encoding objects carrying ids, type_ids, tokens, offsets,
  attention_mask, special_tokens_mask and word_ids. The ids are a small fraction of that. The
  existing tokenize_wikipedia.py batches 20,000 documents at a time, which materialises all of
  that at once. This batches far smaller and converts each Encoding straight to a uint16 array,
  so the heavy objects are released immediately instead of accumulating.

Output is written incrementally, so a kill does not lose the whole run (onboarding section 7:
"a long job that only writes at the end produces nothing if killed" - hours lost twice).
"""
import argparse
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from data.clean_wikipedia import iter_articles
from tools.memtrack import report

TOKENIZER = Path("tokenizer/tokenizer_v2.json")
OUT_DIR = Path("data/pretrain")

# Small on purpose. 20,000 documents of Encoding objects is hundreds of MB of metadata that is
# thrown away one line later.
DOCS_PER_BATCH = 1_000


def tokenize_file(tokenizer, source, destination, token_target, endoftext_id):
    if not source.exists():
        raise SystemExit(f"missing {source} - run data/build_pretrain_corpus.py first")

    print(f"{source.name} -> {destination.name} (target {token_target:,} tokens)")
    total_tokens = 0
    total_docs = 0
    batch = []
    started = time.time()

    def flush(batch, total_tokens):
        encodings = tokenizer.encode_batch(batch)
        pieces = []
        for encoding in encodings:
            pieces.append(np.asarray(encoding.ids, dtype=np.uint16))
            pieces.append(np.array([endoftext_id], dtype=np.uint16))
        del encodings                       # release offsets/masks before concatenating
        chunk = np.concatenate(pieces)
        del pieces
        remaining = token_target - total_tokens
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        return chunk

    with open(destination, "wb") as out:
        for document in iter_articles(source):
            if total_tokens >= token_target:
                break
            batch.append(document)
            if len(batch) < DOCS_PER_BATCH:
                continue

            chunk = flush(batch, total_tokens)
            chunk.tofile(out)
            total_tokens += len(chunk)
            total_docs += len(batch)
            batch.clear()
            del chunk

            if total_docs % (DOCS_PER_BATCH * 200) == 0:
                rate = total_tokens / (time.time() - started)
                pct = total_tokens / token_target * 100
                print(f"  {total_docs:,} docs -> {total_tokens:,} tokens ({pct:.1f}%) "
                      f"| {rate/1000:.0f}k tok/s")

        if batch and total_tokens < token_target:
            chunk = flush(batch, total_tokens)
            chunk.tofile(out)
            total_tokens += len(chunk)
            total_docs += len(batch)

    elapsed = time.time() - started
    size_gb = destination.stat().st_size / 1024 ** 3
    print(f"  done: {total_docs:,} documents -> {total_tokens:,} tokens "
          f"({size_gb:.2f} GB) in {elapsed/60:.1f} min")
    report("(after tokenizing)")
    return total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tokens", type=int, default=1_600_000_000)
    parser.add_argument("--validation-tokens", type=int, default=5_000_000)
    args = parser.parse_args()

    if not TOKENIZER.exists():
        raise SystemExit(f"missing {TOKENIZER} - run tokenizer/refit_tokenizer.py first")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    vocab_size = tokenizer.get_vocab_size()
    endoftext_id = tokenizer.token_to_id("<|endoftext|>")
    if vocab_size > np.iinfo(np.uint16).max:
        raise SystemExit(f"vocab {vocab_size} does not fit in uint16")
    print(f"tokenizer {TOKENIZER} | vocab {vocab_size:,} | <|endoftext|> id {endoftext_id}")

    tokenize_file(tokenizer, OUT_DIR / "pretrain_train.txt", OUT_DIR / "pretrain_train.bin",
                  args.train_tokens, endoftext_id)
    print()
    tokenize_file(tokenizer, OUT_DIR / "pretrain_validation.txt",
                  OUT_DIR / "pretrain_validation.bin", args.validation_tokens, endoftext_id)


if __name__ == "__main__":
    main()
