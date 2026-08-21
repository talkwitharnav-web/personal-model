import joblib
import re

from orchestrator import config


class Classifier:
    """Decides what kind of request this is, so the pipeline knows how to handle it."""

    def __init__(self, embedder):
        if not config.ROUTER_PATH.exists():
            raise SystemExit(
                f"No router at {config.ROUTER_PATH} - run `python -m router.train_router` first."
            )
        self.embedder = embedder
        self.model = joblib.load(config.ROUTER_PATH)

    def classify(self, question):
        vector = self.embedder.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        )
        probabilities = self.model.predict_proba(vector)[0]
        best = int(probabilities.argmax())
        return self.model.classes_[best], float(probabilities[best])

    def wants_retrieval(self, category, confidence):
        confident = confidence >= config.ROUTER_CONFIDENCE
        return not (confident and category in config.NO_RETRIEVAL_CATEGORIES)


# Questions about the CONVERSATION rather than the world. "what's my dog's name?" was routed
# `instruction 0.87` -> retrieval, which pulled three unrelated Wikipedia passages about dog
# breeds and names; they filled the context and buried the one place the answer actually was,
# the previous turn. Wikipedia cannot know the user's dog, so retrieval can only do harm here.
#
# Deliberately narrow: this only applies when there IS prior conversation, so a first-message
# question is never affected, and it keys on self-reference plus a short question. "what causes
# my headaches" is long enough and factual enough to keep its passages.
PERSONAL_PATTERNS = re.compile(
    r"\b(my|mine|i|me|myself|we|our)\b.*\b(name|called|said|told|mentioned|earlier|"
    r"before|just now|dog|cat|pet|favourite|favorite)\b"
    r"|\b(what|who|which|when|where)\b.*\bdid i\b"
    r"|\byou (just )?(said|told|mentioned)\b",
    re.I,
)


def is_personal_context(question):
    """True if the answer would come from the conversation, not from Wikipedia."""
    return bool(PERSONAL_PATTERNS.search(question)) and len(question.split()) <= 12


def normalise_shouting(question):
    """Lowercase input that is mostly capitals, before anything else sees it.

    The tokenizer was fitted on clean Wikipedia, where sustained capitals barely occur, so an
    ALL CAPS sentence shatters into fragments. Independently verified 2026-08-20 on 6 questions
    x 2 seeds: raw caps 4/12, normalised 6/12, lowercase 6/12 - it fixed 2 of 12 paired cases and
    regressed none, and normalised output matched lowercase output in 12/12. Mean token count fell
    16.7 -> 7.8 with routes and retrieved passages identical, which isolates generator tokenisation
    as the mechanism rather than retrieval.

    ⚠ The gain is real but MODEST - only one of six question types was actually rescued. An earlier
    version of this docstring claimed caps "break answers", which the 4/12 baseline does not
    support.

    ⚠ Requires at least three alphabetic words. Verification found that terse acronym and proper
    noun input ("NASA HQ", "UN HQ") crossed the old letter-ratio threshold and got lowercased,
    which is a semantic loss rather than a fix. Real shouting is a sentence; a terse acronym query
    is one or two words, so the word count separates them. The cost is that a two-word shout like
    "HELLO THERE" is left alone - a much cheaper mistake than mangling a factual query.
    """
    letters = [c for c in question if c.isalpha()]
    if len(letters) < 5:
        return question
    if len([w for w in question.split() if any(c.isalpha() for c in w)]) < 3:
        return question
    upper_share = sum(c.isupper() for c in letters) / len(letters)
    return question.lower() if upper_share > 0.7 else question
