"""Teach the model to read messy human input: typos, shorthand, no capitals, no punctuation.

The problem, measured in `tools/typo_robustness_probe.py`: `tokenizer_v2` was fitted on clean
Wikipedia, so "like" is one token and "loke" is [' l', 'oke'] - two fragments whose embeddings were
learned independently and which nothing has ever connected. Real input destroys real answers:

    "What is the capital of France?"   -> "The capital of France is Paris ..."
    "What is teh capitol of Frnace?"   -> "a former Roman Catholic bishop"

The tokenizer cannot be changed - it is baked into a 30-hour pretrain. What CAN be changed is the
fine-tune data, and the fix is the standard one: show the model the same question written badly and
the same correct answer, so the fragments learn to point at the same meaning.

THE CRITICAL RULE: corrupt the USER turn only, never the ASSISTANT turn. The goal is "messy input,
clean correct output". Corrupting the answers would teach the model to *write* badly, which is the
opposite of what anyone wants, and packing masks everything except the assistant span anyway - so a
corrupted answer would be corrupting the only thing that carries gradient.

Corruptions are weighted toward what people actually type. Pure random character noise is easy to
generate and teaches the wrong distribution: real typos are adjacent keys, transpositions, doubled
and dropped letters, missing apostrophes, and deliberate shorthand.

Usage:
    python -m data.augment_typos --datasets no_robots,oasst1,soda,definitions --rate 0.35
"""
import argparse
import random
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Keys physically next to each other on QWERTY - the single most common real typo.
NEIGHBOURS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr", "f": "drtgvc",
    "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn", "k": "jiolm", "l": "kop",
    "m": "njk", "n": "bhjm", "o": "iklp", "p": "ol", "q": "wa", "r": "edft", "s": "awedxz",
    "t": "rfgy", "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx",
}

# Deliberate shorthand, not mistakes. Wikipedia never writes these, so the model has no
# representation of them at all unless they are put in front of it.
SHORTHAND = {
    "you": "u", "your": "ur", "you're": "ur", "are": "r", "please": "pls", "thanks": "thx",
    "thank you": "ty", "because": "bc", "before": "b4", "people": "ppl", "really": "rly",
    "probably": "prob", "definitely": "def", "about": "abt", "with": "w/", "without": "w/o",
    "something": "smth", "someone": "some1", "tonight": "2nite", "tomorrow": "tmrw",
    "i don't know": "idk", "to be honest": "tbh", "in my opinion": "imo",
    "by the way": "btw", "as soon as possible": "asap", "oh my god": "omg",
    "laughing": "lmao", "very funny": "lol", "what": "wat", "okay": "ok", "though": "tho",
}

PROTECTED = {"a", "i", "an", "the", "is", "it", "to", "of", "in", "on", "at"}


def swap_adjacent_key(word, rng):
    if len(word) < 3:
        return word
    i = rng.randrange(1, len(word))
    letter = word[i].lower()
    if letter not in NEIGHBOURS:
        return word
    return word[:i] + rng.choice(NEIGHBOURS[letter]) + word[i + 1:]


def transpose(word, rng):
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def drop_letter(word, rng):
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word))
    return word[:i] + word[i + 1:]


def double_letter(word, rng):
    if len(word) < 3:
        return word
    i = rng.randrange(1, len(word))
    return word[:i] + word[i] + word[i:]


CORRUPTIONS = [swap_adjacent_key, transpose, drop_letter, double_letter]


def corrupt_words(text, rng, word_rate):
    """Typo a share of the long words, leaving short function words alone."""
    def replace(match):
        word = match.group(0)
        if len(word) < 4 or word.lower() in PROTECTED or rng.random() > word_rate:
            return word
        return rng.choice(CORRUPTIONS)(word, rng)
    return re.sub(r"[A-Za-z']+", replace, text)


def apply_shorthand(text, rng, chance):
    lowered = text
    for phrase, short in sorted(SHORTHAND.items(), key=lambda kv: -len(kv[0])):
        if rng.random() < chance:
            lowered = re.sub(rf"\b{re.escape(phrase)}\b", short, lowered, flags=re.I)
    return lowered


def messify(text, rng):
    """One realistic 'typed in a hurry' rewrite of a user turn."""
    style = rng.random()
    out = text
    if style < 0.35:                       # mostly typos
        out = corrupt_words(out, rng, word_rate=0.25)
    elif style < 0.65:                     # mostly shorthand
        out = apply_shorthand(out, rng, chance=0.5)
    else:                                  # both
        out = apply_shorthand(out, rng, chance=0.35)
        out = corrupt_words(out, rng, word_rate=0.15)

    if rng.random() < 0.45:                # no capitals
        out = out.lower()
    if rng.random() < 0.35:                # no closing punctuation
        out = out.rstrip(".!?").rstrip()
    if rng.random() < 0.15:                # doubled punctuation
        out = out.rstrip() + rng.choice(["!!", "??", "...", "!!!"])
    if rng.random() < 0.20:                # dropped apostrophes
        out = out.replace("'", "")
    return out


def messify_user_turns(block, rng):
    """Rewrite only the User: lines. Assistant lines are the training target - never touch them."""
    lines = block.split("\n")
    changed = False
    for i, line in enumerate(lines):
        if line.startswith("User:"):
            body = line[len("User:"):].strip()
            if not body:
                continue
            messy = messify(body, rng)
            if messy != body:
                lines[i] = f"User: {messy}"
                changed = True
    return ("\n".join(lines), changed)


def main():
    parser = argparse.ArgumentParser()
    # Every dataset whose User turn is something a person actually types. Chat data alone is not
    # enough: if only conversational rows are augmented, the model learns "messy input -> chat
    # answer" and a misspelt QUESTION still breaks question answering, which is most of the mix.
    # Datasets excluded on purpose:
    #   replay      - raw pretrain prose, no user turn at all
    #   xsum        - the "question" is a news article, not something anyone types
    #   proofwriter - synthetic logic statements; corrupting them changes the logic
    #   hellaswag / piqa / socialiqa / commonsenseqa / reclor / logiqa / race - the user turn is a
    #                 passage plus fixed options, so corruption would damage the options too
    parser.add_argument("--datasets",
                        default="no_robots,oasst1,soda,personachat,definitions,"
                                "yesno,squad,sciq,boolq,gsm8k_clean")
    parser.add_argument("--rate", type=float, default=0.35,
                        help="share of examples to emit a messy COPY of")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--out", default="data/typo_aug")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for split in ("train", "validation"):
        target = out_dir / f"{split}.txt"
        collected = []
        for name in args.datasets.split(","):
            path = Path("data") / name / f"{split}.txt"
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for block in text.split("<|endoftext|>"):
                block = block.strip()
                if not block or "\nAssistant:" not in block:
                    continue
                if rng.random() > args.rate:
                    continue
                messy, changed = messify_user_turns(block, rng)
                if not changed:
                    continue
                collected.append(messy)
        # Shuffle before writing. Written in source order, the file's first N blocks are all one
        # dataset - and every auditor that reads a head instead of a sample then measures that one
        # dataset while appearing to measure the file. That bug has already been hit twice today.
        rng.shuffle(collected)
        with open(target, "w", encoding="utf-8") as sink:
            for block in collected:
                sink.write(block + "\n<|endoftext|>\n")
        print(f"Done: {target} - {len(collected):,} messy copies")

    print("\nEach copy keeps its ORIGINAL assistant answer, so the model learns")
    print("'messy input -> clean correct output'. Add 'typo_aug' to dataset_names in")
    print("data/pack_finetune_datasets.py and data/combine_finetune_datasets.py to use it.")
    print("\nExamples:")
    demo = ["What is the capital of France?", "Can you help me with this please?",
            "I really like apples and oranges.", "Thank you very much, that was helpful!"]
    for line in demo:
        print(f"    {line}")
        print(f"      -> {messify(line, rng)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
