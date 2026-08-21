"""Measure what vocab size actually costs: 16k vs 32k BPE on the REAL corpus.

The open question in onboarding.md section 10 step 3 is "keep 32k or drop to 16k". The tradeoff is
a straight swap:

  smaller vocab -> fewer embedding parameters, smaller logits tensor
  smaller vocab -> WORSE compression, so the same text becomes MORE tokens

Nobody can eyeball the second number, so this trains both tokenizers on the same sample of
data/wikipedia/train.txt and measures tokens-per-byte on a held-out slice taken from a
different part of the file.

Byte discipline (see onboarding section 7): the file is read in BINARY and all offsets stay in
bytes end to end. Scanning in text mode while counting byte lengths silently truncated 17,162
articles once.
"""
import time
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

CORPUS = Path("data/wikipedia/train.txt")
WORK = Path("tests/vocab_probe")
TRAIN_SAMPLE_BYTES = 300 * 1024 * 1024   # 300 MB to fit BPE training in RAM comfortably
HELDOUT_BYTES = 20 * 1024 * 1024
HELDOUT_OFFSET = 6 * 1024 * 1024 * 1024  # 6 GB in - disjoint from the training sample
SPECIALS = ["<unk>", "<pad>", "<|endoftext|>"]
VOCABS = [16000, 32000]

EMBED_DIM = 640
CONTEXT = 768
BATCH = 16
NON_EMBED_PARAMS = 59_083_520   # v2 total 79,563,520 minus the 32000x640 tied embedding


def slice_bytes(path, offset, length):
    """Read a byte range and trim to whole lines so no token is cut in half."""
    with open(path, "rb") as handle:
        handle.seek(offset)
        raw = handle.read(length)
    if offset > 0:
        cut = raw.find(b"\n")
        raw = raw[cut + 1:] if cut != -1 else raw
    cut = raw.rfind(b"\n")
    if cut != -1:
        raw = raw[:cut + 1]
    return raw


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    train_path = WORK / "sample_train.txt"
    heldout_path = WORK / "heldout.txt"

    if not train_path.exists():
        print(f"extracting {TRAIN_SAMPLE_BYTES/1024/1024:.0f} MB training sample ...")
        train_path.write_bytes(slice_bytes(CORPUS, 0, TRAIN_SAMPLE_BYTES))
    if not heldout_path.exists():
        print(f"extracting {HELDOUT_BYTES/1024/1024:.0f} MB held-out slice from offset "
              f"{HELDOUT_OFFSET/1024/1024/1024:.0f} GB ...")
        heldout_path.write_bytes(slice_bytes(CORPUS, HELDOUT_OFFSET, HELDOUT_BYTES))

    heldout_raw = heldout_path.read_bytes()
    heldout_text = heldout_raw.decode("utf-8", errors="ignore")
    heldout_bytes = len(heldout_raw)
    heldout_words = len(heldout_text.split())
    print(f"held-out: {heldout_bytes:,} bytes, {heldout_words:,} whitespace words\n")

    results = {}
    for vocab in VOCABS:
        out = WORK / f"tok_{vocab}.json"
        if not out.exists():
            print(f"training BPE vocab={vocab:,} ...")
            started = time.time()
            tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            tokenizer.pre_tokenizer = ByteLevel()
            tokenizer.train([str(train_path)],
                            BpeTrainer(vocab_size=vocab, special_tokens=SPECIALS, min_frequency=2))
            tokenizer.save(str(out))
            print(f"  done in {(time.time()-started)/60:.1f} min")

        tokenizer = Tokenizer.from_file(str(out))
        # Encode in chunks: one 20 MB encode call builds a huge offsets/tokens structure.
        total = 0
        chunk = []
        size = 0
        for line in heldout_text.splitlines(keepends=True):
            chunk.append(line)
            size += len(line)
            if size > 1_000_000:
                total += len(tokenizer.encode("".join(chunk)).ids)
                chunk, size = [], 0
        if chunk:
            total += len(tokenizer.encode("".join(chunk)).ids)
        results[vocab] = total
        print(f"vocab {vocab:,}: {total:,} tokens on the held-out slice "
              f"| {heldout_bytes/total:.3f} bytes/token | {total/heldout_words:.4f} tokens/word")

    small, large = VOCABS[0], VOCABS[1]
    inflation = results[small] / results[large]
    print("\n" + "=" * 72)
    print(f"TOKEN INFLATION going {large:,} -> {small:,}: x{inflation:.4f} "
          f"({(inflation-1)*100:+.2f}% more tokens for the same text)")

    for vocab in VOCABS:
        embed = vocab * EMBED_DIM
        logits_mb = BATCH * CONTEXT * vocab * 2 / 1024 / 1024   # fp16 logits tensor
        print(f"\nvocab {vocab:,}")
        print(f"  embedding params (tied)   {embed:,}")
        print(f"  total model params        {NON_EMBED_PARAMS + embed:,}")
        print(f"  embedding share of model  {embed/(NON_EMBED_PARAMS+embed)*100:.1f}%")
        print(f"  fp16 logits per fwd pass  {logits_mb:.0f} MB  (batch {BATCH} x {CONTEXT})")

    # The decisive number: same TEXT budget, not same TOKEN budget.
    print("\n" + "=" * 72)
    budget_32k = 1_600_000_000
    equivalent_16k = budget_32k * inflation
    print(f"Chinchilla-ish budget of {budget_32k/1e9:.1f}B tokens at 32k vocab covers some amount of text.")
    print(f"Covering that SAME text at 16k vocab costs {equivalent_16k/1e9:.3f}B tokens "
          f"({(inflation-1)*100:+.2f}%).")
    print(f"At ~14,700 tok/s that is {budget_32k/14700/3600:.1f} h vs "
          f"{equivalent_16k/14700/3600:.1f} h -> {(equivalent_16k-budget_32k)/14700/3600:+.1f} h")
    print(f"\nContext window reach: 768 tokens spans "
          f"{768*heldout_bytes/results[large]:.0f} bytes at 32k vs "
          f"{768*heldout_bytes/results[small]:.0f} bytes at 16k of raw text.")


if __name__ == "__main__":
    main()
