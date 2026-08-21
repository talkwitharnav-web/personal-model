"""Why does the model answer "Yes" to everything?

From Dark's testing transcript, all of these got "Yes":

    what does hot mean?              -> Yes
    how old do kids tend to be?      -> Yes
    are fishes and cats the same?    -> Yes
    my mother is so damn greedy      -> Yes
    what's my dog's name?            -> Yes
    do you like self-harm?           -> Yes! I love being a good father.

The hypothesis: masking makes this far worse than the example counts suggest. Only the assistant
span carries gradient, and for a yes/no row that span is a SINGLE token. So while yes/no and BoolQ
are only about 4.6% of examples, they may be a much larger share of the tokens the model is
actually trained to produce - and crucially, of the tokens that appear immediately after
"Assistant:", which is exactly the position where an answer starts.

If "Yes" dominates that position, then "Yes" is simply the model's prior for "what do I say next",
and it will fall back to it whenever it is uncertain. That is a data-balance problem, not a
mysterious behavioural one.

Measured on the packed data the model actually trained on, so nothing is inferred from example
counts alone.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from tokenizers import Tokenizer

PACKED = Path("data/finetune_packed")


def main():
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer_v2.json")
    ids = np.load(PACKED / "train_ids.npy", mmap_mode="r")
    labels = np.load(PACKED / "train_labels.npy", mmap_mode="r")

    rows = min(30000, len(ids))
    print(f"reading {rows:,} of {len(ids):,} packed rows\n")

    trained = Counter()          # every token that carries gradient
    first_tokens = Counter()     # the FIRST token of each assistant span
    spans = 0
    single_token_spans = 0

    for start in range(0, rows, 2048):
        chunk_ids = np.asarray(ids[start:start + 2048])
        chunk_labels = np.asarray(labels[start:start + 2048])
        for row_ids, row_labels in zip(chunk_ids, chunk_labels):
            active = row_labels != -100
            trained.update(row_labels[active].tolist())
            # A span starts wherever a trained position follows an untrained one.
            previous = False
            length = 0
            for value, is_active in zip(row_labels, active):
                if is_active and not previous:
                    first_tokens[int(value)] += 1
                    spans += 1
                    length = 1
                elif is_active:
                    length += 1
                elif previous:
                    if length == 1:
                        single_token_spans += 1
                    length = 0
                previous = is_active

    total_trained = sum(trained.values())
    print(f"gradient-carrying tokens: {total_trained:,}")
    print(f"assistant spans:          {spans:,}")
    print(f"  of which ONE token long: {single_token_spans:,} "
          f"({100*single_token_spans/max(spans,1):.1f}%)")

    def token_id(text):
        encoded = tokenizer.encode(text).ids
        return encoded[0] if encoded else None

    watch = {}
    for word in ("Yes", "No", " Yes", " No"):
        tid = token_id(word)
        if tid is not None:
            watch.setdefault(tid, word)

    print("\nshare of ALL trained tokens:")
    for tid, word in watch.items():
        count = trained.get(tid, 0)
        print(f"  {word!r:<7} id {tid:<6} {count:>9,}  {100*count/max(total_trained,1):5.2f}%")

    print("\nshare of the FIRST token of an answer (the position that decides the reply):")
    for tid, word in watch.items():
        count = first_tokens.get(tid, 0)
        print(f"  {word!r:<7} id {tid:<6} {count:>9,}  {100*count/max(spans,1):5.2f}%")

    print("\nmost common FIRST tokens overall:")
    for tid, count in first_tokens.most_common(12):
        text = tokenizer.decode([tid])
        print(f"  {text!r:<16} {count:>8,}  {100*count/max(spans,1):5.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
