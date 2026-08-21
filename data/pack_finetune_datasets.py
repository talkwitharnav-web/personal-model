"""Step 6: tokenize the fine-tune mix with tokenizer_v2 AND pack it, eliminating padding waste.

THE PROBLEM (onboarding section 10.1): 71.3% of every fine-tune step was padding. Mean real
length is 220.4 tokens of 768, median 164. Every fine-tune run has cost roughly 3.5x what it
needed to, and the padded positions contribute nothing - they are masked to -100, so they produce
no gradient at all.

THE FIX: stop storing one example per 768-token row. Concatenate examples end to end and cut at
768, so almost every position is real. The last row of the file is the only padded one.

This also retokenizes with tokenizer_v2 (+9.42% compression), which the fine-tune data needs
anyway - every existing *_ids.npy was built with the v1 tokenizer.

⚠ CROSS-EXAMPLE ATTENTION. Packing means a later example can attend to an earlier one inside the
same 768-token window, because the attention is causal but not segmented. This is standard
practice and the examples are separated by <|endoftext|>, which the model is explicitly trained to
treat as a boundary (onboarding section 7: the stop token was never trained in v1, and fixing that
was one of the expensive bugs). Truly isolating them needs block-diagonal attention masks, which
is a model change, not a data change - noted here so the trade-off is a decision rather than an
accident.

⚠ THE REPLAY TRAP (onboarding section 10). Replay rows are raw pretraining passages with no
"\\nAssistant:" marker, so every existing tokenize script silently DISCARDS them (answer_start is
None -> return None). They need the opposite treatment: no masking at all, train on every token.
NO_MASK_DATASETS below is that flag.
"""
import argparse
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from tools.memtrack import report

TOKENIZER = Path("tokenizer/tokenizer_v2.json")
OUT_DIR = Path("data/finetune_packed")
CONTEXT = 768

dataset_names = [
    "commonsenseqa", "sciq", "piqa", "boolq", "hellaswag", "socialiqa",
    "logiqa", "reclor", "squad", "race", "xsum", "oasst1", "personachat",
    "proofwriter", "definitions", "yesno",
    # GSM8K is in the v2 mix. Section 10 reverses the v1 exclusion: the objection was about
    # disrupting a nearly-finished run, which cannot apply to a model trained from zero. Uses
    # the cleaned copy - the shipped file ends every answer with GSM8K's "#### <number>" grader
    # marker, which would teach the model to emit it.
    "gsm8k_clean",
    # SODA and No Robots go INTO the mix, not a third stage - a third stage reintroduces
    # catastrophic forgetting (section 10). SODA buys conversational register, No Robots buys
    # varied human-written instructions.
    "soda", "no_robots",
    # 5% replay: raw pretrain passages, so old knowledge is actively reinforced rather than
    # passively preserved. Trains on every token - see NO_MASK_DATASETS below.
    "replay",
    # Messy-input copies of the sets above: typos, shorthand, no capitals, no punctuation. The
    # tokenizer was fitted on clean Wikipedia, so "loke" becomes [' l', 'oke'] - fragments with no
    # learned relationship to " like". Nothing else in the mix ever shows the model that real
    # people type badly. Only the USER turn is corrupted; answers and retrieved contexts are
    # byte-identical to the originals (tests/verify_typo_augmentation.py).
    "typo_aug",
    # Bare greetings. Measured 2026-08-19: only 2,951 of 85,251 conversations in the mix opened
    # with one, and the distribution decided behaviour exactly - 'hello' (593 examples) worked,
    # 'hey' (104) produced mid-conversation filler, 'yo' (~0) was echoed back.
    "greetings",
    # Commands, negation and false premises. The mix answers a QUESTION correctly and fails the
    # same fact as a COMMAND, because opening-verb coverage collapses: explain 4,240 / give 4,157
    # / describe 3,817 all work, while list 141 / name 137 / provide 129 / show 39 / compare 37
    # do not. Also carries the only decline examples in the mix - just 1.1% of all answers push
    # back on anything, which is why "When did Einstein invent the telephone?" produced "1876".
    "instruction_shapes",
]

# XSum was 157,923 examples - 27.6% of the whole mix - so more than a quarter of
# training was "summarise a news article", crowding out the question answering.
#
# typo_aug is capped for a related reason: at the full 90,675 it would be the second-largest set in
# the mix, and every one of its rows carries an answer that ALSO appears in its clean source, so
# those answers would be trained twice. 60,000 is ample to learn that "wat" means "what" without
# giving duplicated targets that much weight.
MAX_PER_DATASET = {"xsum": 50_000, "typo_aug": 60_000}

# Datasets whose rows carry no "\nAssistant:" marker and must be trained on in full.
NO_MASK_DATASETS = {"replay"}

SEED = 42
ROWS_PER_FLUSH = 2048


def count_documents(path):
    """Count non-empty documents by streaming, without holding the file."""
    total = 0
    buffer = ""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        while True:
            data = handle.read(8_000_000)
            if not data:
                break
            buffer += data
            parts = buffer.split("<|endoftext|>")
            buffer = parts.pop()
            total += sum(1 for part in parts if part.strip())
    if buffer.strip():
        total += 1
    return total


def iter_examples(path, limit, rng):
    """Stream example texts one at a time, applying the cap with the documented rng.

    The earlier version did path.read_text() and then .split(), which for xsum means a 466 MB
    file becomes a 466 MB string plus a list of 204,000 more strings - well over a gigabyte for
    one dataset, on a machine also running a 30-hour training job. Streaming holds one document.

    Costs one extra pass to count documents first, because the cap has to draw from a known
    total to keep selecting the same rows as before.
    """
    keep = None
    if limit:
        total = count_documents(path)
        if total > limit:
            keep = set(np.random.default_rng(0).choice(total, limit, replace=False).tolist())

    index = 0
    buffer = ""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        while True:
            data = handle.read(8_000_000)
            if not data:
                break
            buffer += data
            parts = buffer.split("<|endoftext|>")
            buffer = parts.pop()
            for part in parts:
                if not part.strip():
                    continue
                if keep is None or index in keep:
                    yield part
                index += 1
    if buffer.strip():
        if keep is None or index in keep:
            yield buffer


def encode_example(tokenizer, marker, endoftext, text, mask):
    """Return (ids, labels) with no padding, or None if unusable."""
    ids = tokenizer.encode(text).ids + [endoftext]
    if len(ids) > CONTEXT:
        return None

    if not mask:
        # Replay rows: every token is a target.
        return ids, list(ids)

    answer_start = None
    for i in range(len(ids) - len(marker) + 1):
        if ids[i:i + len(marker)] == marker:
            answer_start = i + len(marker)      # last occurrence, as the v1 scripts did
    if answer_start is None:
        return None

    labels = list(ids)
    for i in range(answer_start):
        labels[i] = -100
    return ids, labels


def build(split_name, out_dir):
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    endoftext = tokenizer.token_to_id("<|endoftext|>")
    # BPE is context sensitive: encoding "Assistant:" alone differs from the same text mid
    # document. Encode "\nAssistant:" and drop the first token to get a marker that matches
    # real data (onboarding section 5).
    marker = tokenizer.encode("\nAssistant:").ids[1:]

    rng = random.Random(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_ids = out_dir / f"{split_name}_examples_ids.raw"
    scratch_labels = out_dir / f"{split_name}_examples_labels.raw"

    # PHASE 1: tokenize each dataset in turn, streaming encoded examples straight to scratch
    # files and keeping only their lengths. Holding every example's TEXT in memory instead
    # would be ~1 GB+ for the train split, on a machine that is simultaneously running a
    # 30-hour training job - and Python dies silently when RAM runs short (section 2).
    lengths = []
    skipped_long = skipped_no_marker = 0
    with open(scratch_ids, "wb") as ids_file, open(scratch_labels, "wb") as labels_file:
        for name in dataset_names:
            path = Path(f"data/{name}/{split_name}.txt")
            if not path.exists():
                print(f"  WARNING: missing {path}, skipping")
                continue
            cap = MAX_PER_DATASET.get(name) if split_name == "train" else None
            kept = 0
            for text in iter_examples(path, cap, rng):
                encoded = encode_example(tokenizer, marker, endoftext, text,
                                         mask=name not in NO_MASK_DATASETS)
                if encoded is None:
                    # Distinguish the two reasons: a missing marker in a masked dataset is a
                    # data problem worth seeing, an over-long example is expected.
                    if len(tokenizer.encode(text).ids) + 1 > CONTEXT:
                        skipped_long += 1
                    else:
                        skipped_no_marker += 1
                    continue
                ids, labels = encoded
                np.asarray(ids, dtype=np.int32).tofile(ids_file)
                np.asarray(labels, dtype=np.int32).tofile(labels_file)
                lengths.append(len(ids))
                kept += 1
            print(f"  {name}: {kept:,}")

    lengths = np.asarray(lengths, dtype=np.int64)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    total_examples = len(lengths)
    real_tokens = int(offsets[-1])
    print(f"  total examples: {total_examples:,} | real tokens {real_tokens:,}")

    # Shuffle BEFORE packing so a single 768-token window spans several datasets, the same
    # reason the v1 mix was blended rather than trained sequentially. Only the ORDER is
    # shuffled - an index of 571k int64s is ~4.5 MB, against ~1 GB for the examples themselves.
    order = np.random.default_rng(SEED).permutation(total_examples)

    src_ids = np.memmap(scratch_ids, dtype=np.int32, mode="r", shape=(real_tokens,))
    src_labels = np.memmap(scratch_labels, dtype=np.int32, mode="r", shape=(real_tokens,))

    raw_ids = out_dir / f"{split_name}_ids.raw"
    raw_labels = out_dir / f"{split_name}_labels.raw"

    # Numpy buffers, not Python lists. Building rows with .tolist() creates a Python int object
    # per token - 118 million of them over the train split - and CPython's allocator does not
    # hand the arenas back, so peak private memory hit 4,951 MB. With fixed numpy buffers the
    # hot loop allocates nothing. An example is at most CONTEXT tokens and the leftover is under
    # CONTEXT, so 2x CONTEXT always suffices.
    buffer_ids = np.empty(CONTEXT * 2, dtype=np.int32)
    buffer_labels = np.empty(CONTEXT * 2, dtype=np.int32)
    buffered = 0
    pending_ids = np.empty((ROWS_PER_FLUSH, CONTEXT), dtype=np.int32)
    pending_labels = np.empty((ROWS_PER_FLUSH, CONTEXT), dtype=np.int32)
    pending = 0
    rows = 0

    with open(raw_ids, "wb") as ids_file, open(raw_labels, "wb") as labels_file:
        for index in order:
            start, stop = int(offsets[index]), int(offsets[index + 1])
            length = stop - start
            buffer_ids[buffered:buffered + length] = src_ids[start:stop]
            buffer_labels[buffered:buffered + length] = src_labels[start:stop]
            buffered += length

            while buffered >= CONTEXT:
                pending_ids[pending] = buffer_ids[:CONTEXT]
                pending_labels[pending] = buffer_labels[:CONTEXT]
                pending += 1
                rows += 1
                remainder = buffered - CONTEXT
                buffer_ids[:remainder] = buffer_ids[CONTEXT:buffered]
                buffer_labels[:remainder] = buffer_labels[CONTEXT:buffered]
                buffered = remainder
                if pending == ROWS_PER_FLUSH:
                    pending_ids.tofile(ids_file)
                    pending_labels.tofile(labels_file)
                    pending = 0

        # One final partial row, padded. This is the ONLY padding in the whole file.
        if buffered:
            pending_ids[pending, :buffered] = buffer_ids[:buffered]
            pending_ids[pending, buffered:] = endoftext
            pending_labels[pending, :buffered] = buffer_labels[:buffered]
            pending_labels[pending, buffered:] = -100
            pending += 1
            rows += 1
        if pending:
            pending_ids[:pending].tofile(ids_file)
            pending_labels[:pending].tofile(labels_file)

    del src_ids, src_labels
    scratch_ids.unlink()
    scratch_labels.unlink()

    # Wrap the raw streams as proper .npy files now that the row count is known. Copying via a
    # memmap keeps this bounded instead of loading 500 MB arrays.
    for raw_path, out_name in ((raw_ids, f"{split_name}_ids.npy"),
                               (raw_labels, f"{split_name}_labels.npy")):
        source = np.memmap(raw_path, dtype=np.int32, mode="r", shape=(rows, CONTEXT))
        destination = np.lib.format.open_memmap(out_dir / out_name, mode="w+",
                                                dtype=np.int32, shape=(rows, CONTEXT))
        for start in range(0, rows, ROWS_PER_FLUSH):
            stop = min(start + ROWS_PER_FLUSH, rows)
            destination[start:stop] = source[start:stop]
        destination.flush()
        del destination, source
        raw_path.unlink()

    padded_rows_equivalent = total_examples
    total_positions = rows * CONTEXT
    print(f"\n  packed {split_name}: {rows:,} rows x {CONTEXT} = {total_positions:,} positions")
    print(f"    real tokens {real_tokens:,} -> {real_tokens/max(total_positions,1)*100:.1f}% "
          f"of positions are real (v1 was 28.7%)")
    print(f"    unpadded equivalent: {padded_rows_equivalent:,} padded rows -> {rows:,} packed "
          f"= {padded_rows_equivalent/max(rows,1):.2f}x fewer steps")
    print(f"    skipped: {skipped_long:,} too long, {skipped_no_marker:,} no marker")
    report(f"(after {split_name})")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--splits", nargs="+", default=["train", "validation"],
                        choices=["train", "validation"])
    args = parser.parse_args()
    if not TOKENIZER.exists():
        raise SystemExit(f"missing {TOKENIZER} - run tokenizer/refit_tokenizer.py first")
    for split_name in args.splits:
        print(f"=== {split_name} ===")
        build(split_name, Path(args.out))


if __name__ == "__main__":
    main()
