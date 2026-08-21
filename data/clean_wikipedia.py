"""Step 4a: clean Wikipedia for the v2 pretraining corpus.

Rules written against MEASURED junk rates, not guesses - see tools/audit_wikipedia_junk.py.
On a 419 MB / 159,373-article sample:

  disambiguation    4.60% of articles      -> drop the article (link lists, no prose)
  empty parens      8.27% of articles      -> REPAIR the line, do not drop it
  stripped values   2.84% of articles      -> drop the sentence (the value is gone for good)
  bare headings     4.85% of LINES         -> drop the line
  bibliography      0.02% of lines         -> drop the line
  raw URLs          0.09% of lines         -> drop the line
  true redirects    0.00%                  -> the HF dump already excludes them

THE KEY DECISION IS REPAIR VS DELETE. "Krilatica () is a village in the municipality of Kratovo,
North Macedonia." is a perfectly good sentence with two stray characters in it; the dump strips
template values and leaves the brackets behind (onboarding section 11). Deleting those articles
would throw away 8% of the corpus to remove 2 characters. So empty brackets are repaired in place.

Where the value itself is gone - "He is the eldest son of , who served as..." or "The wingspan is
about ." - the fact cannot be recovered, and training on it teaches the model to emit sentences
that trail off. Those sentences are dropped, but the rest of the article is kept.

Writes to NEW files. data/wikipedia/train.txt is left alone: it is the source for
rag/wikipedia_passages.txt and for the LR-sweep .bin files, and a corpus and the artifacts built
from it are a matched set.
"""
import argparse
import re
import time
from pathlib import Path

from tools.memtrack import report

SOURCE_DIR = Path("data/wikipedia")
OUT_DIR = Path("data/pretrain")

# --- article-level rejects -------------------------------------------------
DISAMBIGUATION = re.compile(r"\bmay (?:also )?refer to\s*:|\bis a disambiguation page", re.I)
LIST_TITLE = re.compile(r"^\s*(?:List|Index|Outline|Glossary|Timeline) of\b", re.I)

# --- line-level rejects ----------------------------------------------------
BARE_HEADING = re.compile(
    r"^\s*(?:See also|References|External links|Further reading|Notes|Sources|Bibliography|"
    r"Citations|Footnotes|Gallery|Contents|External sources)\s*$", re.I)
BIBLIOGRAPHY = re.compile(r"\bISBN\b|\bdoi:|\bOCLC\b|\bJSTOR\b|\barXiv\b|"
                          r"Archived from the original|Retrieved \d|Retrieved on")
# Markup and URLs are STRIPPED, not used to reject the line. Measured on 49,421 articles:
# rejecting any line containing markup threw away real prose - "Born in White Plains, New York,
# Roberto went to the NYU's Tisch School of the Arts..." was deleted for a stray tag near the end.
# Deleting a sentence to remove a two-character tag is the wrong trade.
FILE_LINK = re.compile(r"\[\[(?:File|Image|Media)\s*:[^\]]*\]\]", re.I)
REF_TAG = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/?>|</ref>", re.I | re.S)
TEMPLATE_OPEN = re.compile(r"\{\{\s*[A-Za-z0-9 _-]*\|?")
WIKILINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
URL = re.compile(r"https?://\S+|www\.\S+")
DASH_RUN = re.compile(r"(?:\s*[-\u2013\u2014]\s*){2,}$")
# Removing a URL or template can leave a sentence hanging: "visit the site at". Training on
# that teaches the model to trail off mid-thought, which is the same defect the stripped-value
# rule exists to prevent - so trailing function words are peeled off until the line ends on
# something substantive.
DANGLING_END = re.compile(
    r"[\s,;:]*\b(?:of|at|in|on|by|to|for|from|with|the|a|an|and|or|as|is|was|are|were|"
    r"that|this|which|near|about)\s*$", re.I)

# After stripping, a line can be left as "- -" or "}}". Five letters is enough to keep genuine
# short headings ("Career", "Legacy") while dropping residue.
MIN_LINE_LETTERS = 5
LETTERS = re.compile(r"[^A-Za-z]")

# --- sentence-level rejects: the value was stripped and cannot be recovered -
STRIPPED_VALUE = re.compile(
    r"\b(?:of|at|in|is|was|are|were|about|approximately|around|reaching|measuring|"
    r"between|from|to|by|with|near|over|under)\s+[,.]")

# --- repairs ---------------------------------------------------------------
EMPTY_BRACKETS = re.compile(r"\s*\(\s*[,;:]?\s*\)|\s*\[\s*[,;:]?\s*\]")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
MULTI_SPACE = re.compile(r"[ \t]{2,}")
MULTI_COMMA = re.compile(r",\s*,+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

MIN_ARTICLE_CHARS = 200


def clean_line(line):
    """Repair what is repairable, drop what is not. Returns '' to drop the line."""
    if BARE_HEADING.match(line) or BIBLIOGRAPHY.search(line):
        return ""

    # Strip markup and URLs, keeping whatever prose surrounded them.
    line = FILE_LINK.sub(" ", line)
    line = REF_TAG.sub(" ", line)
    line = WIKILINK.sub(r"\1", line)
    line = TEMPLATE_OPEN.sub(" ", line).replace("}}", " ")
    line = URL.sub(" ", line)

    line = EMPTY_BRACKETS.sub("", line)

    if STRIPPED_VALUE.search(line):
        # Surgical: keep the sentences that survived, drop only the broken ones.
        kept = [s for s in SENTENCE_SPLIT.split(line) if not STRIPPED_VALUE.search(s)]
        line = " ".join(kept)

    line = MULTI_COMMA.sub(",", line)
    line = SPACE_BEFORE_PUNCT.sub(r"\1", line)
    line = MULTI_SPACE.sub(" ", line)
    line = DASH_RUN.sub("", line).strip()

    previous = None
    while previous != line:                    # "... site at the" needs two passes
        previous = line
        line = DANGLING_END.sub("", line).strip()

    if len(LETTERS.sub("", line)) < MIN_LINE_LETTERS:
        return ""
    return line


def clean_article(article):
    head = article[:300]
    if DISAMBIGUATION.search(head) or LIST_TITLE.match(article.lstrip()):
        return None, "disambiguation_or_list"

    lines = [clean_line(l) for l in article.split("\n")]
    lines = [l for l in lines if l]
    if not lines:
        return None, "emptied"

    cleaned = "\n".join(lines)
    if len(cleaned) < MIN_ARTICLE_CHARS:
        # Only near-empty stubs. NOT a general length filter - onboarding section 7 records that
        # filtering retrieval passages on length broke the Hamlet answer (a 55-char list entry).
        # This is the pretraining corpus, where a 90-character fragment teaches nothing.
        return None, "too_short"
    return cleaned, None


def iter_articles(path, read_bytes=8_000_000):
    """Stream on the <|endoftext|> boundary. Never decode a multi-GB file as one string.

    8 MB chunks, not 64: split() briefly holds a list of every article in the chunk alongside
    the chunk itself, so the read size sets the peak. 8 MB keeps this stage around 25 MB instead
    of ~200 MB. RAM is this machine's real bottleneck and Python dies silently when it runs out
    (onboarding section 2).
    """
    buffer = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        while True:
            data = handle.read(read_bytes)
            if not data:
                break
            buffer += data
            parts = buffer.split("<|endoftext|>")
            # The tail is a partial article; it becomes the next buffer and stays small.
            buffer = parts.pop()
            for part in parts:
                article = part.strip()
                if article:
                    yield article
            del parts
    if buffer.strip():
        yield buffer.strip()


def clean_file(source, destination, target_bytes):
    stats = {"in": 0, "out": 0, "disambiguation_or_list": 0, "too_short": 0, "emptied": 0}
    bytes_in = bytes_out = 0
    started = time.time()

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Written incrementally: a job that only writes at the end produces nothing if killed
    # (onboarding section 7 - hours lost twice to exactly that).
    with open(destination, "w", encoding="utf-8") as out:
        for article in iter_articles(source):
            stats["in"] += 1
            bytes_in += len(article)
            cleaned, reason = clean_article(article)
            if cleaned is None:
                stats[reason] += 1
                continue
            out.write(cleaned + "\n<|endoftext|>\n")
            stats["out"] += 1
            bytes_out += len(cleaned)

            if stats["in"] % 200_000 == 0:
                rate = bytes_in / (time.time() - started) / 1024 / 1024
                print(f"  {stats['in']:,} read -> {stats['out']:,} kept | "
                      f"{bytes_out/1024**3:.2f} GB written | {rate:.0f} MB/s read")
            if target_bytes and bytes_out >= target_bytes:
                print(f"  target {target_bytes/1024**3:.2f} GB reached, stopping early")
                break

    elapsed = time.time() - started
    print(f"\n{source.name} -> {destination.name}  ({elapsed/60:.1f} min)")
    print(f"  articles  {stats['in']:,} read, {stats['out']:,} kept "
          f"({stats['out']/max(1,stats['in'])*100:.1f}%)")
    print(f"    dropped disambiguation/list  {stats['disambiguation_or_list']:,}")
    print(f"    dropped too short (<{MIN_ARTICLE_CHARS})     {stats['too_short']:,}")
    print(f"    dropped emptied by cleaning  {stats['emptied']:,}")
    print(f"  bytes     {bytes_in/1024**3:.2f} GB in -> {bytes_out/1024**3:.2f} GB out "
          f"({bytes_out/max(1,bytes_in)*100:.1f}% retained)")
    report("(after cleaning)")
    return stats


def main():
    parser = argparse.ArgumentParser()
    # ~1.28B Wikipedia tokens at 4.165 bytes/token = 5.33 GB; 7.5 GB is a deliberate overshoot
    # so the mixing step can pick freely without running short.
    parser.add_argument("--target-gb", type=float, default=7.5)
    args = parser.parse_args()

    print("=== cleaning Wikipedia for the v2 pretrain corpus ===")
    clean_file(SOURCE_DIR / "train.txt", OUT_DIR / "wikipedia_clean.txt",
               int(args.target_gb * 1024 ** 3))
    print()
    clean_file(SOURCE_DIR / "validation.txt", OUT_DIR / "wikipedia_clean_validation.txt", 0)


if __name__ == "__main__":
    main()
