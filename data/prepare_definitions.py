# Builds question/answer pairs from Wikipedia's opening sentences, which are almost
# always "X is a Y that Z". The model currently answers "what is a tesla coil" with
# "A Tesla coil" - it learned to return the noun phrase before "is" rather than read
# the question. The cure is CONTRASTIVE examples: the same passage asked three
# different ways, so the only way to be right is to actually read the question.
#
# v2: the first version put ONE sentence in the Context, but at answer time retrieval
# hands over three long passages - so the skill never transferred. Each example now
# sits in a realistic multi-passage Context. The distractors are real definition
# sentences chosen to be *plausible*: they prefer to share a word with the subject,
# and for when/who questions they prefer to carry a rival year or a rival creator.
# Filler that were obviously irrelevant would only teach it to discard nonsense.
import random
import re
from pathlib import Path

WIKIPEDIA_PATH = "data/wikipedia/train.txt"
output_dir = Path("data/definitions")

TARGET_EXAMPLES = 70_000     # ~12% of the combined mix; enough to teach, not to dominate
VALIDATION_FRACTION = 0.02
# See the write loop below: the share of definitions asked with no Context block, so the model
# learns to define from knowledge and not only to extract from a supplied passage.
NO_CONTEXT_FRACTION = 0.25
MAX_ARTICLES = 400_000
READ_BYTES = 50_000_000

# How many passages sit in the Context. Retrieval serves TOP_K=3, but it must still
# cope when the filter leaves fewer, so keep a share of the easy single-passage case.
DISTRACTOR_WEIGHTS = {0: 0.2, 1: 0.3, 2: 0.5}
NEAR_MISS_CHANCE = 0.7       # how often a distractor is drawn from the shared-word pool
STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
             "is", "was", "are", "were", "that", "this", "with", "from", "by"}

# "A Tesla coil is a resonant transformer circuit invented by Nikola Tesla in 1891."
#   subject = "A Tesla coil"   predicate = "a resonant transformer circuit invented by..."
DEFINITION = re.compile(r"^(.{3,60}?)\s+(?:is|was|are|were)\s+(.{25,300}?)\.\s")
CREATOR = re.compile(
    r"\b(?:invented|founded|created|written|developed|discovered|designed|composed|"
    r"directed|produced|built|established)\s+by\s+([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,3})"
)
YEAR = re.compile(r"\b(1[5-9]\d{2}|20[0-2]\d)\b")
# Only ask "when" if the sentence actually dates an event. Otherwise the first year
# is usually a person's birth date, giving nonsense like "When was <person> made?".
DATED_EVENT = re.compile(
    r"\b(?:founded|established|released|published|invented|created|formed|opened|"
    r"introduced|launched|completed|built|premiered|adopted)\b[^.]{0,40}?\b(1[5-9]\d{2}|20[0-2]\d)\b"
)
# Subjects carry trailing brackets - "Strange Sisters ()", "X (born 13 October 1980)".
PARENTHETICAL = re.compile(r"\s*\([^)]*\)")

# "written by American diplomat William Russell" captured just "American" - the
# capitalised word after "by" is often a nationality, not the creator.
DESCRIPTORS = {
    "American", "British", "English", "French", "German", "Japanese", "Italian",
    "Spanish", "Canadian", "Australian", "Russian", "Chinese", "Indian", "Dutch",
    "Swedish", "Irish", "Scottish", "Welsh", "Polish", "Danish", "Norwegian",
    "Greek", "Turkish", "Brazilian", "Mexican", "Korean", "Swiss", "Austrian",
    "Belgian", "Portuguese", "Finnish", "Czech", "Hungarian", "Romanian",
    "Professor", "Dr", "Sir", "Lord", "Saint", "King", "Queen", "President",
}

# ⚠ THE SAME MISTAKE, A THIRD TIME - caught before training rather than after.
# definitions v1 matched the FORMAT but not the context LENGTH. yesno v1 matched format and
# length but not the question PHRASING, and needed 53 templates before the skill generalised
# beyond the exact wordings it was taught. This file had ELEVEN, so it had the identical latent
# defect: the model would learn "answer a definition question" only at these exact shapes.
#
# Confirmed in the pretrained model, which knows "Paris is the capital and most populous city of
# France" (Wikipedia's exact opening formula) but answers "the province of Quebec, Canada" to
# "Paris is the capital". A fact learned at one phrasing is not a fact, it is a memorised string.
#
# So: many templates per fact type, covering the axes a real user actually varies.
WHAT_TEMPLATES = [
    "What is {s}?", "What are {s}?", "Define {s}.", "What does {s} mean?",
    "Can you explain {s}?", "Tell me about {s}.", "What exactly is {s}?",
    "I've never heard of {s}, what is it?", "What is meant by {s}?",
    "Could you describe {s}?", "How would you describe {s}?", "What sort of thing is {s}?",
    "Explain what {s} is.", "Give me a definition of {s}.", "What's {s}?",
    "In simple terms, what is {s}?", "Do you know what {s} is?",
    "What kind of thing is {s}?", "Describe {s}.", "What is {s} exactly?",
]
WHO_TEMPLATES = [
    "Who made {s}?", "Who created {s}?", "Who is {s} attributed to?", "Who invented {s}?",
    "Who developed {s}?", "Who came up with {s}?", "Who was {s} created by?",
    "Do you know who made {s}?", "Who is credited with {s}?", "Who is behind {s}?",
    "Who was responsible for {s}?", "By whom was {s} created?", "Who built {s}?",
    "Who founded {s}?", "Can you tell me who created {s}?",
]
WHEN_TEMPLATES = [
    "When was {s} made?", "In what year did {s} originate?", "When did {s} begin?",
    "When was {s} created?", "What year was {s} made?", "When did {s} first appear?",
    "How long ago was {s} made?", "Do you know when {s} was created?",
    "When does {s} date from?", "What year did {s} start?", "When was {s} founded?",
    "When did {s} come about?", "Around when was {s} created?",
]

# Real users type "what is a tesla coil" - no capital, no question mark. A dataset written
# entirely in tidy prose teaches the model that tidy prose is part of the question.
STYLE_WEIGHTS = {"plain": 0.78, "no_mark": 0.07, "lower_start": 0.05,
                 "lower_all": 0.06, "lower_all_no_mark": 0.04}


def restyle(question, style):
    if style in ("lower_all", "lower_all_no_mark"):
        question = question.lower()
    elif style == "lower_start" and question:
        question = question[0].lower() + question[1:]
    if style in ("no_mark", "lower_all_no_mark"):
        question = question.rstrip("?.").rstrip()
    return question


# 92.8% of answers were bare lowercase fragments ("a species of sea snail..."), so the dataset
# taught one output SHAPE as well as one question shape. A user asking "what is a tesla coil"
# should get a usable answer whether the model replies with a phrase or a sentence.
#
# ⚠ The obvious full-sentence form - "A Tesla coil is a resonant transformer circuit..." - is
# NOT used, because it restates the subject from the question and would reward exactly the
# subject-echo shortcut §9 spent two days killing and which regressed once already. Pronoun
# forms give sentence structure without ever paying the model to copy the question.
ANSWER_FORMS = {
    "what": [("{a}", 6), ("It is {a}.", 2), ("It's {a}.", 1), ("This is {a}.", 1)],
    "who": [("{a}", 6), ("It was created by {a}.", 2), ("That was {a}.", 1)],
    "when": [("{a}", 6), ("It dates from {a}.", 2), ("That was in {a}.", 1)],
}


def shape_answer(answer, kind, rng):
    forms = ANSWER_FORMS.get(kind)
    if not forms:
        return answer
    template = rng.choices([f for f, _ in forms], weights=[w for _, w in forms])[0]
    if template == "{a}":
        return answer
    # Only wrap a fragment; an answer that is already a sentence reads wrong inside one.
    if answer[:1].isupper() and answer.rstrip().endswith("."):
        return answer
    return template.format(a=answer.rstrip("."))

# Openings that are lists, disambiguation pages or headings rather than definitions.
REJECT = re.compile(r"^(?:this |the following|list of|index of|outline of)", re.I)


def iter_articles(path):
    buffer = ""
    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(READ_BYTES)
            if not chunk:
                break
            buffer += chunk
            parts = buffer.split("<|endoftext|>")
            buffer = parts.pop()
            for article in parts:
                yield article


def first_sentence(article):
    text = " ".join(article.strip().split())
    match = re.search(r"^(.{40,400}?\.)\s", text + " ")
    return match.group(1) if match else None


def content_words(text):
    """Words worth matching on, so 'Tesla coil' can find 'Tesla Motors'."""
    return {w for w in re.findall(r"[A-Za-z][\w'-]{2,}", text.lower())
            if w not in STOPWORDS}


def build_examples(sentence, rng):
    """One passage, up to three different questions - that is what forces it to read."""
    definition = DEFINITION.match(sentence + " ")
    if not definition:
        return []
    subject, predicate = definition.group(1).strip(), definition.group(2).strip()

    subject = PARENTHETICAL.sub("", subject).strip(" ,;:")
    if not (3 <= len(subject) <= 60) or "(" in subject or ")" in subject:
        return []
    # "The Almeida Theatre opened in 1980, is a ..." - the subject swallowed the verb
    # phrase, giving "When did The Almeida Theatre opened in 1980 begin?", which hands
    # the model the answer in the question.
    if YEAR.search(subject):
        return []
    # A comma means the subject swallowed a leading clause, e.g.
    # "In computer science, a parallel external memory model".
    if "," in subject or REJECT.match(subject) or len(subject.split()) > 8:
        return []

    # One style per example, applied to whichever template was drawn. Chosen here rather than
    # per template so casing never correlates with the kind of question being asked.
    style = rng.choices(list(STYLE_WEIGHTS), weights=list(STYLE_WEIGHTS.values()))[0]

    examples = [(rng.choice(WHAT_TEMPLATES).format(s=subject), predicate, "what")]

    creator = CREATOR.search(sentence)
    if creator:
        words = [w for w in creator.group(1).split() if w not in DESCRIPTORS]
        if words:
            examples.append(
                (rng.choice(WHO_TEMPLATES).format(s=subject), " ".join(words), "who"))

    dated = DATED_EVENT.search(sentence)
    if dated:
        examples.append(
            (rng.choice(WHEN_TEMPLATES).format(s=subject), dated.group(1), "when"))

    return [(restyle(q, style), shape_answer(a, kind, rng), kind, subject)
            for q, a, kind in examples
            if a.lower() not in q.lower()]


def pick_distractors(row, sentences, by_word, rival, rng):
    """Plausible wrong passages: same register, ideally overlapping or rival facts."""
    _, answer, kind, subject, source = row
    wanted = rng.choices(list(DISTRACTOR_WEIGHTS), weights=list(DISTRACTOR_WEIGHTS.values()))[0]
    if not wanted:
        return []

    near = set()
    for word in content_words(subject):
        near.update(by_word.get(word, ()))
    # Converted once: rebuilding this tuple inside the draw loop costs O(len(near)) per
    # attempt, and a common word can put thousands of sentences in here.
    near = tuple(near)
    # A rival year / rival "by <Name>" makes when- and who-questions genuinely hard.
    themed = rival.get(kind, ())

    chosen, seen = [], {source}
    for _ in range(wanted * 12):
        if len(chosen) == wanted:
            break
        if near and rng.random() < NEAR_MISS_CHANCE:
            candidate = sentences[rng.choice(near)]
        elif themed and rng.random() < 0.5:
            candidate = sentences[rng.choice(themed)]
        else:
            candidate = sentences[rng.randrange(len(sentences))]
        # Never let a distractor also contain the answer - that makes the label a lie
        # about which passage was read, and stops the example teaching anything.
        if candidate in seen or answer.lower() in candidate.lower():
            continue
        if subject.lower() in candidate.lower():
            continue
        seen.add(candidate)
        chosen.append(candidate)
    return chosen


def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    rows = []
    seen = set()

    for count, article in enumerate(iter_articles(WIKIPEDIA_PATH)):
        if count >= MAX_ARTICLES or len(rows) >= TARGET_EXAMPLES:
            break
        sentence = first_sentence(article)
        if sentence:
            for question, answer, kind, subject in build_examples(sentence, rng):
                if (question, answer) in seen:
                    continue
                seen.add((question, answer))
                rows.append((question, answer, kind, subject, sentence))
        if count % 50_000 == 0 and count:
            print(f"  {count:,} articles read -> {len(rows):,} examples")

    sentences = sorted({row[4] for row in rows})
    by_word, rival = {}, {"who": [], "when": []}
    for i, sentence in enumerate(sentences):
        for word in content_words(sentence[:80]):
            by_word.setdefault(word, []).append(i)
        if CREATOR.search(sentence):
            rival["who"].append(i)
        if YEAR.search(sentence):
            rival["when"].append(i)
    print(f"  distractor pool: {len(sentences):,} sentences, "
          f"{len(rival['who']):,} with a creator, {len(rival['when']):,} with a year")

    rng.shuffle(rows)
    split = int(len(rows) * (1 - VALIDATION_FRACTION))
    widths = []

    for name, subset in (("train", rows[:split]), ("validation", rows[split:])):
        path = output_dir / f"{name}.txt"
        bare_count = 0
        with open(path, "w", encoding="utf-8") as f:
            for row in subset:
                question, answer, source = row[0], row[1], row[4]
                # A quarter of definitions are asked with no passage at all, so the model
                # practises knowing as well as extracting. Every example used to hand over a
                # context containing the answer verbatim, which trains copying: measured, the
                # answer was findable in the context 61% of the time. With no context the model
                # produced "The Tesla coil is a coil that is a coil" - it had never been asked
                # to define anything unaided. Kept to a quarter because RAG supplies the passage
                # at serving time, so extraction must stay the dominant signal.
                if rng.random() < NO_CONTEXT_FRACTION:
                    bare_count += 1
                    f.write(f"User: {question}\nAssistant: {answer}\n<|endoftext|>\n")
                    continue
                passages = pick_distractors(row, sentences, by_word, rival, rng)
                widths.append(len(passages) + 1)
                # The true passage must not sit in a fixed slot, or position becomes
                # the shortcut instead of the noun phrase.
                passages.insert(rng.randint(0, len(passages)), source)
                context = " ".join(passages)
                f.write(f"User: {question}\n\nContext: {context}\n"
                        f"Assistant: {answer}\n<|endoftext|>\n")
        print(f"Done: {path} - {len(subset):,} examples "
              f"({bare_count:,} with no context, {bare_count / max(len(subset), 1):.0%})")

    spread = {n: widths.count(n) / len(widths) for n in sorted(set(widths))}
    print("  passages per context: "
          + ", ".join(f"{n}x {share:.0%}" for n, share in spread.items()))


if __name__ == "__main__":
    main()
