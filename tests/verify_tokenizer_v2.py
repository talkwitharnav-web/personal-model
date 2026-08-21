"""Validate the refitted tokenizer before anything is retokenized with it.

Retokenizing the corpus is hours of work and it invalidates every .bin and .npy in the repo, so
the tokenizer gets checked first. These are the checks that fail SILENTLY - a broken tokenizer
still encodes and still trains, just worse.
"""
from pathlib import Path

from tokenizers import Tokenizer

V1 = Path("tokenizer/tokenizer.json")
V2 = Path("tokenizer/tokenizer_v2.json")
HELDOUT = Path("data/wikipedia/validation.txt")
HELDOUT_BYTES = 8 * 1024 * 1024

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}  {detail}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def encode_in_chunks(tokenizer, text, limit=1_000_000):
    """One huge encode() call builds an enormous offsets structure - chunk it."""
    total, chunk, size = 0, [], 0
    for line in text.splitlines(keepends=True):
        chunk.append(line)
        size += len(line)
        if size > limit:
            total += len(tokenizer.encode("".join(chunk)).ids)
            chunk, size = [], 0
    if chunk:
        total += len(tokenizer.encode("".join(chunk)).ids)
    return total


def main():
    if not V2.exists():
        raise SystemExit(f"missing {V2} - run tokenizer/refit_tokenizer.py first")
    v2 = Tokenizer.from_file(str(V2))
    v1 = Tokenizer.from_file(str(V1))

    print("structure - must match v1 or every downstream script breaks")
    print("-" * 58)
    check("vocab size is exactly 32,000", v2.get_vocab_size() == 32000,
          f"{v2.get_vocab_size():,}")
    for expected_id, token in enumerate(["<unk>", "<pad>", "<|endoftext|>"]):
        actual = v2.token_to_id(token)
        check(f"{token} keeps id {expected_id}", actual == expected_id, f"got {actual}")
    check("decoder is configured", v2.decode(v2.encode("hello world").ids).strip() != "",
          "ByteLevel decoder present")

    print("\nround trip - lossless apart from the documented prefix space")
    print("-" * 58)
    # ByteLevel is configured with add_prefix_space=True (in v1 too), so encode() prepends a
    # space and decode() gives it back. That is NOT data loss and NOT a regression - but the
    # naive assertion decoded == text fails on it, which looks alarming. The real requirements
    # are (a) content survives, and (b) v2 behaves exactly like v1, since every downstream
    # script was written against v1's behaviour.
    samples = [
        "The Tesla coil is a resonant transformer circuit invented by Nikola Tesla in 1891.",
        "User: who wrote Hamlet?\nAssistant: William Shakespeare.",
        "Numbers 1081, 3.14159 and symbols: %, &, --- \u2014 \u00e9\u00e8\u00ea, \u4e2d\u6587",
        "   leading and trailing whitespace   ",
    ]
    for text in samples:
        decoded_v2 = v2.decode(v2.encode(text).ids)
        decoded_v1 = v1.decode(v1.encode(text).ids)
        check(f"content survives: {text[:30]!r}", decoded_v2.lstrip(" ") == text.lstrip(" "),
              "" if decoded_v2.lstrip(" ") == text.lstrip(" ") else f"got {decoded_v2[:44]!r}")
        check(f"matches v1 behaviour: {text[:30]!r}",
              (decoded_v2 == text) == (decoded_v1 == text),
              f"v2 exact={decoded_v2 == text}, v1 exact={decoded_v1 == text}")

    print("\nthe marker gotcha (onboarding section 5)")
    print("-" * 58)
    # BPE is context sensitive: encoding "Assistant:" alone gives different ids than the same
    # text mid-document. The fix used everywhere is encode("\nAssistant:").ids[1:]. If that
    # subsequence does not appear in a real formatted row, EVERY tokenize script silently
    # discards every example (answer_start is None -> return None) and the run trains on nothing.
    marker = v2.encode("\nAssistant:").ids[1:]
    document = ("User: What is photosynthesis?\nContext: Plants convert light into chemical "
                "energy.\nAssistant: Plants convert light into chemical energy.")
    ids = v2.encode(document).ids
    found = any(ids[i:i + len(marker)] == marker for i in range(len(ids) - len(marker) + 1))
    check("\\nAssistant: marker (ids[1:]) appears in a real row", found,
          f"marker {marker}")

    naive = v2.encode("Assistant:").ids
    naive_found = any(ids[i:i + len(naive)] == naive for i in range(len(ids) - len(naive) + 1))
    check("and the naive encode('Assistant:') still does NOT match "
          "(so the ids[1:] trick is still required)", not naive_found, f"naive {naive}")

    print("\ncompression on held-out text - the whole point of refitting")
    print("-" * 58)
    raw = HELDOUT.read_bytes()[:HELDOUT_BYTES]
    cut = raw.rfind(b"\n")
    raw = raw[:cut + 1] if cut != -1 else raw
    text = raw.decode("utf-8", errors="ignore")
    n_bytes = len(raw)

    v1_tokens = encode_in_chunks(v1, text)
    v2_tokens = encode_in_chunks(v2, text)
    v1_bpt = n_bytes / v1_tokens
    v2_bpt = n_bytes / v2_tokens
    improvement = (v1_tokens - v2_tokens) / v1_tokens * 100
    print(f"  v1 (TinyStories+WikiText fit): {v1_tokens:,} tokens | {v1_bpt:.3f} bytes/token")
    print(f"  v2 (real corpus fit):          {v2_tokens:,} tokens | {v2_bpt:.3f} bytes/token")
    check("v2 compresses Wikipedia better than v1", v2_tokens < v1_tokens,
          f"{improvement:+.2f}% fewer tokens for the same text")

    unk_id = v2.token_to_id("<unk>")
    unk_count = sum(1 for i in v2.encode(text[:400_000]).ids if i == unk_id)
    check("no <unk> explosion on real text", unk_count == 0, f"{unk_count} unks")

    print(f"\n{passed} passed, {failed} failed")
    if v2_tokens < v1_tokens:
        print(f"\n{improvement:.2f}% fewer tokens means the same 1.6B-token budget now covers "
              f"{improvement:.2f}% more text at identical cost.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
