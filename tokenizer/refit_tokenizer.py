"""Step 3 of the rebuild: refit the 32k BPE tokenizer on the REAL corpus.

The v1 tokenizer (tokenizer/tokenizer.json) was fitted on TinyStories + a small WikiText sample,
so its merges are tuned for children's stories while the model trains on Wikipedia. Vocab size was
measured and confirmed at 32,000 (tests/tokenizer_vocab_probe.py) - the size was never the problem,
the training data was.

MUST match v1's structure exactly or every downstream script breaks:
  - ByteLevel pre-tokenizer AND ByteLevel decoder (v1 has both)
  - specials in this exact order, so <unk>=0, <pad>=1, <|endoftext|>=2
    (data/tokenize_*.py append pad_id, and onboarding section 7 refers to endoftext as id 2)

Writes to a NEW path. tokenizer/tokenizer.json still serves combined_best_model.pt through chat.py
and must not be touched - a checkpoint and its tokenizer are a matched pair.

Byte discipline (onboarding section 7): the corpus is read in BINARY and every offset stays in
bytes. Scanning in text mode while counting byte lengths silently truncated 17,162 articles once.
"""
import time
from pathlib import Path

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

CORPUS = Path("data/wikipedia/train.txt")
OUT = Path("tokenizer/tokenizer_v2.json")
SAMPLE = Path("tokenizer/refit_sample.txt")

VOCAB_SIZE = 32000
SPECIALS = ["<unk>", "<pad>", "<|endoftext|>"]   # order fixes the ids - do not reorder
MIN_FREQUENCY = 2

# 2 GB drawn from 8 evenly spaced points rather than the head of the file. The corpus was
# shuffled at download so the head is probably fine, but "probably" is not a reason to sample
# lazily, and spreading the reads costs nothing.
SAMPLE_TOTAL = 2 * 1024 * 1024 * 1024
CHUNKS = 8


def build_sample():
    if SAMPLE.exists():
        print(f"reusing existing sample {SAMPLE} ({SAMPLE.stat().st_size/1024**3:.2f} GB)")
        return
    corpus_size = CORPUS.stat().st_size
    per_chunk = SAMPLE_TOTAL // CHUNKS
    stride = corpus_size // CHUNKS
    print(f"corpus {corpus_size/1024**3:.1f} GB -> sampling {CHUNKS} x "
          f"{per_chunk/1024**2:.0f} MB (stride {stride/1024**3:.2f} GB)")

    written = 0
    with open(CORPUS, "rb") as source, open(SAMPLE, "wb") as sink:
        for index in range(CHUNKS):
            source.seek(index * stride)
            raw = source.read(per_chunk)
            if index * stride > 0:
                cut = raw.find(b"\n")            # drop the half line we landed in the middle of
                raw = raw[cut + 1:] if cut != -1 else raw
            cut = raw.rfind(b"\n")               # and the half line at the end
            if cut != -1:
                raw = raw[:cut + 1]
            sink.write(raw)
            written += len(raw)
            print(f"  chunk {index+1}/{CHUNKS} at {index*stride/1024**3:6.2f} GB "
                  f"-> {len(raw)/1024**2:.0f} MB (total {written/1024**3:.2f} GB)")


def main():
    if not CORPUS.exists():
        raise SystemExit(f"missing corpus: {CORPUS}")
    build_sample()

    print(f"\ntraining BPE vocab={VOCAB_SIZE:,} on {SAMPLE.stat().st_size/1024**3:.2f} GB ...")
    started = time.time()
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()     # v1 has this; without it decode() emits 'Ġ' junk
    tokenizer.train([str(SAMPLE)],
                    BpeTrainer(vocab_size=VOCAB_SIZE,
                               special_tokens=SPECIALS,
                               min_frequency=MIN_FREQUENCY))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUT))
    print(f"trained in {(time.time()-started)/60:.1f} min -> {OUT}")

    for token in SPECIALS:
        print(f"  {token:<15} id {tokenizer.token_to_id(token)}")
    print(f"  vocab size     {tokenizer.get_vocab_size():,}")

    SAMPLE.unlink(missing_ok=True)
    print(f"removed {SAMPLE}")
    print("\nNOTE: this invalidates every tokenized artifact (data/**/*.bin, *.npy). They must be "
          "regenerated before training against this tokenizer. Existing checkpoints stay paired "
          "with tokenizer/tokenizer.json.")


if __name__ == "__main__":
    main()
