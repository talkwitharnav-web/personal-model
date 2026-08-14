import os
import re
from pathlib import Path
from spellchecker import SpellChecker

# Casual/slang words real people use that a formal dictionary doesn't know -
# don't "correct" these away.
SAFE_WORDS = {
    "gonna", "wanna", "gotta", "yeah", "ok", "okay", "hmm", "um", "uh", "huh",
    "yep", "nah", "dunno", "kinda", "lemme", "gimme", "ya", "alright", "ok's",
    "pingpong", "singsong",
    # British spellings and real words the US dictionary wrongly flags
    "centre", "colour", "favour", "theatre", "cheque", "organise", "realise",
    "recognise", "apologise", "webcam", "rearview", "makable", "bitesize",
    "annum", "intra", "fibre", "yarning", "memorise", "criticise",
    "admin", "delicates", "saimin",
}

spell = SpellChecker()
spell.word_frequency.load_words(SAFE_WORDS)

# The raw dataset splits contractions with spaces around the apostrophe,
# e.g. "didn \u2019 t" instead of "didn't" - rejoin those first, or the
# spellchecker sees "didn" as a standalone (mis-)word and corrupts it.
CONTRACTION_GAP = re.compile(r"(\w+)\s*[\u2019']\s*(t|s|m|re|ve|ll|d)\b", re.IGNORECASE)

def rejoin_contractions(line):
    return CONTRACTION_GAP.sub(r"\1'\2", line)

word_pattern = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")
correction_cache = {}

def get_correction(lower):
    if lower in correction_cache:
        return correction_cache[lower]
    correction = spell.correction(lower)
    correction_cache[lower] = correction
    return correction

def fix_line(line, stats):
    line = rejoin_contractions(line)

    def fix_word(match):
        word = match.group(0)
        lower = word.lower()

        # Contractions (didn't, wasn't) - never split or "correct" these.
        if "'" in word:
            return word

        # Only touch plain lowercase words - capitalized words are usually
        # names, which a general dictionary will "correct" incorrectly.
        if word != lower or len(word) <= 3 or lower in SAFE_WORDS:
            return word

        if spell.known([lower]):
            return word

        correction = get_correction(lower)
        if correction and correction != lower:
            stats["fixed"] += 1
            stats["examples"].append(f"{lower} -> {correction}")
            return correction
        return word

    return word_pattern.sub(fix_word, line)

def fix_file(path):
    stats = {"fixed": 0, "examples": []}
    temp_path = path.with_suffix(".tmp")

    with open(path, "r", encoding="utf-8") as infile, open(temp_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            outfile.write(fix_line(line, stats))

    os.replace(temp_path, path)
    print(f"{path}: fixed {stats['fixed']} words")
    for example in stats["examples"][:15]:
        print(f"  {example}")

data_dir = Path("data/dailydialog")
fix_file(data_dir / "train.txt")
fix_file(data_dir / "validation.txt")
