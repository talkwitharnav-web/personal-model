"""Contrastive yes/no data.

The model answers yes/no by copying the polarity of the passage instead of reading
the question: given the same Sun passage it said Yes to "is the sun cold" (P=0.799)
AND Yes to "is the sun hot" (P=0.800). BoolQ can never punish that, because every
BoolQ passage appears with exactly one question, so "echo the passage" scores 100%.

So each source sentence here produces a MATCHED PAIR - same context, two questions,
opposite answers. Copying the passage now gets one of the two wrong every time.

v2 (phrasing): the pair structure worked - 7/8 on trained phrasings, including binding
two names present in the same context to the right subject at >88%. But it only worked
AT THE PHRASINGS IT WAS TRAINED ON, because v1 emitted exactly three template strings:

    is coconut high in saturated fat        Yes 0.8312  correct
    is coconut low  in saturated fat        Yes 0.8368  WRONG - polarity copied again
    Did Nikola Tesla invent the Tesla coil? P(Yes)=0.0000 - not recognised as yes/no

The same mistake as definitions v1 (format right, context LENGTH wrong), one axis over.
A generated dataset must vary every axis the user can vary. So now:

  * 53 templates across four fact types instead of three fixed strings - passive AND
    active voice, is/was/are/were/does/do/did/has/have, plural agreement taken from the
    source sentence's own copula, and a share of lowercase / unpunctuated variants.
  * a fourth fact type - ADJECTIVE ANTONYM pairs ("Is Titan large?" -> Yes,
    "Is Titan small?" -> No), which is exactly the shape the coconut probe failed on.

THE ONE RULE THAT MUST NOT BE BROKEN: both halves of a pair use the SAME template and
the SAME casing style. If phrasing correlated with the label at all, the model would
simply learn "template A means yes" - a fresh shortcut, worse than the bug being fixed.
"""
import os
import random
import re
from pathlib import Path

from data.prepare_definitions import (
    DEFINITION, DESCRIPTORS, PARENTHETICAL, REJECT, YEAR,
    content_words, first_sentence, iter_articles,
)

# Overridable so a few hundred pairs can be sampled and eyeballed without a full run.
# The committed defaults are the real ones.
output_dir = Path(os.environ.get("YESNO_OUT_DIR", "data/yesno"))
# The cleaned corpus is strictly better source material for this dataset: it drops the
# disambiguation pages that produced "Did Wu Zetian Wuzhou Wu really found Wuzhou?", and repairs
# the stripped-template artifacts ("Krilatica () is a village") that corrupt subject extraction.
# Falls back to the raw dump if the cleaned file has not been built yet.
_CLEANED = Path("data/pretrain/wikipedia_clean.txt")
SOURCE = Path(os.environ.get(
    "YESNO_SOURCE", _CLEANED if _CLEANED.exists() else "data/wikipedia/train.txt"))
TARGET_PAIRS = int(os.environ.get("YESNO_TARGET_PAIRS", 20_000))   # -> 40,000 examples, 50/50 by construction
MAX_ARTICLES = int(os.environ.get("YESNO_MAX_ARTICLES", 1_200_000))
# Raised from 400,000: requiring the false value to appear in the context rejects pairs that
# would previously have been emitted with an invented rival, so more articles must be read to
# reach TARGET_PAIRS. Reading more is cheap; a shortcut is not.

VALIDATION_FRACTION = 0.02
# Share of PAIRS written with no Context block at all, so the model practises answering from what
# it actually knows as well as from what it is handed.
#
# Why this is needed: every example used to carry a context containing the answer, so the only
# skill taught was reading comprehension. Measured on the step-10,000 checkpoint, P(Yes) sat
# between 0.52 and 0.91 on EVERY bare question - true or false, encyclopedic or everyday - because
# a bare question is a shape the model had never once been trained on. Checkpoints are selected on
# the behavioural harness (section 8), and that harness asks bare questions, so a set that never
# contains one is selecting on a task the model was never taught.
#
# Why it is only a quarter: RAG supplies context at serving time, so reading comprehension must
# stay the dominant signal. Pushing this high would train confident recall of obscure facts the
# model cannot actually know - the "fluently wrong" failure from section 8. Both halves of a pair
# are stripped together, so the contrast that makes the pair work is preserved.
NO_CONTEXT_FRACTION = 0.25
DISTRACTOR_WEIGHTS = {1: 0.15, 2: 0.35, 3: 0.35, 4: 0.15}
# A context of 0 distractors USED to be 20% of the set, and it is structurally incapable of
# holding a rival fact: the only passage present is the one that makes the answer Yes. Every
# such example therefore taught "the wrong name is not in the text", the exact tell measured at
# AUC 0.740. There is now always at least one distractor, and more of them on average, which
# both removes the artifact and raises the share of pairs that survive the in-context rule.
NEAR_MISS_CHANCE = 0.7
THEMED_CHANCE = 0.5          # how often an adjective row pulls a passage stating the antonym
RIVAL_YEAR_SPREAD = 30       # a wrong year stays nearby, so it cannot be spotted by era

# v1 emitted 87% description questions simply because almost every opening sentence has
# a description. Creator and year questions are the ones that proved relation-reading,
# and adjective pairs are the measured hole, so both are drawn ahead of a description
# when the sentence offers them, and description-only sentences are capped.
KIND_WEIGHTS = {"creator": 6.0, "year": 3.0, "adjective": 6.0, "head": 1.0}
HEAD_ONLY_SHARE = 0.65

# Real users type "is the sun cold". Most rows stay well-formed; a fifth do not.
STYLE_WEIGHTS = {"plain": 0.78, "no_mark": 0.06, "lower_start": 0.05,
                 "lower_all": 0.07, "lower_all_no_mark": 0.04}

CREATOR_VERB = re.compile(
    r"\b(invented|founded|created|written|developed|discovered|designed|composed|"
    r"directed|produced|built|established)\s+by\s+([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,3})"
)
DATED_VERB = re.compile(
    r"\b(founded|established|released|published|invented|created|formed|opened|"
    r"introduced|launched|completed|built|premiered|adopted)\b[^.]{0,40}?\b(1[5-9]\d{2}|20[0-2]\d)\b"
)
# prepare_definitions.first_sentence needs 40 characters before it will stop, so a short
# opener ("Wuzhou is a city in Guangxi, China.") swallows the sentence after it and two
# articles end up welded together. Spotting the internal full stop is cheaper than
# changing the shared splitter - abbreviations and initials must not count as one.
ABBREVIATIONS = ("st", "mt", "ft", "dr", "mr", "mrs", "ms", "jr", "sr", "prof", "gen",
                 "col", "capt", "sgt", "rev", "inc", "ltd", "co", "corp", "vs", "etc",
                 "al", "ca", "approx", "no", "vol", "ed", "eds", "pp", "op", "fig")
RUN_ON = re.compile(r"\b(?!(?i:" + "|".join(ABBREVIATIONS) + r")\b)[A-Za-z]{2,}\.\s+[A-Z]")
DISAMBIGUATION = re.compile(r"\bmay (?:also )?refer to\b|\brefers? to:", re.I)
# A predicate cut at a comma must not end mid-clause.
RELATIVE = re.compile(r"\b(?:that|which|who|whom|whose|where|when)\b")
DANGLING = frozenset((
    "and", "or", "but", "of", "in", "on", "at", "to", "by", "for", "from", "with",
    "as", "into", "than", "near", "the", "a", "an", "its", "their", "his", "her"))
# "directed by Danny A." / "a 2012 comedy-drama film directed by Charles S" is the
# splitter stopping inside a name. Half a name is not a fact worth asking about.
INITIAL_END = re.compile(r"(?:^|\s)[A-Z]\.?$")
# "directed by Géza von Bolváry" stops at the lowercase particle, leaving a bare first
# name. A definition head has to be a noun phrase too, or "would it be right to call X
# located in Penobscot?" comes out.
PARTICLES = frozenset((
    "von", "van", "de", "den", "der", "del", "della", "di", "da", "dos", "du", "la",
    "le", "el", "al", "bin", "ibn", "ter", "ten"))
HEAD_START = frozenset((
    "a", "an", "the", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "several", "many", "various", "another", "both", "each"))
# prepare_definitions.REJECT looks for "this " with a trailing space, so a sentence whose
# whole subject is "This" ("This is a list of Polish television events from 2006.") gets
# through and asks "Is This an 1893 painting by Winslow Homer?".
PRONOUN_SUBJECTS = frozenset((
    "this", "that", "these", "those", "it", "he", "she", "they", "there", "here",
    "such", "one", "another", "both", "some", "any", "all", "who", "what", "which"))
HEAD_REJECT = re.compile(r"^(?:an? list\b|an index\b|an outline\b|the following\b)", re.I)
# "Antanas Jaroševičius (1870-1956) was a painter best known for his album published in
# 1912" dates the album, not the painter, and "Was Antanas Jaroševičius published in
# 1912?" is nonsense. A birth date or a life span marks the subject as a person, and a
# person is not released, published, built or created by anyone.
PERSON_HINT = re.compile(
    r"\([^)]*(?:\bborn\b|\bb\.\s*\d{4}|\d{4}\s*[-–—]\s*(?:\d{4}|\))|\bdied\b)", re.I)

# Active voice needs the verb back in its base and simple-past forms: "invented by X"
# -> "Did X invent ...?" / "Was it X who invented ...?". Only these participles are
# ever matched, so the mapping can be exact rather than guessed by suffix rules.
ACTIVE_FORMS = {
    "invented": ("invent", "invented"), "founded": ("found", "founded"),
    "created": ("create", "created"), "written": ("write", "wrote"),
    "developed": ("develop", "developed"), "discovered": ("discover", "discovered"),
    "designed": ("design", "designed"), "composed": ("compose", "composed"),
    "directed": ("direct", "directed"), "produced": ("produce", "produced"),
    "built": ("build", "built"), "established": ("establish", "established"),
}
# "Did the theatre open in 1980?" only works for verbs that can stand without an agent.
# "Did the album release in 2019?" and "Did the club found in 1900?" do not, so the
# active year templates are offered only for these.
INTRANSITIVE_FORMS = {"opened": "open", "premiered": "premiere",
                      "formed": "form", "launched": "launch"}

# Gradable properties, the axis the dataset had nothing on. Both directions are usable:
# whichever one the article states becomes the Yes, its partner becomes the No.
ANTONYM_PAIRS = [
    ("large", "small"), ("big", "small"), ("huge", "tiny"), ("tall", "short"),
    ("long", "short"), ("wide", "narrow"), ("deep", "shallow"), ("thick", "thin"),
    ("heavy", "light"), ("dense", "sparse"), ("high", "low"), ("rich", "poor"),
    ("abundant", "scarce"), ("hot", "cold"), ("warm", "cool"), ("wet", "dry"),
    ("humid", "arid"), ("hard", "soft"), ("smooth", "rough"), ("bright", "dim"),
    ("loud", "quiet"), ("fast", "slow"), ("ancient", "modern"), ("old", "new"),
    ("traditional", "modern"), ("natural", "artificial"), ("natural", "synthetic"),
    ("common", "rare"), ("popular", "obscure"), ("famous", "obscure"),
    ("urban", "rural"), ("coastal", "inland"), ("northern", "southern"),
    ("eastern", "western"), ("tropical", "polar"), ("mountainous", "flat"),
    ("fictional", "real"), ("extinct", "alive"), ("wild", "domesticated"),
    ("carnivorous", "herbivorous"), ("edible", "inedible"), ("poisonous", "harmless"),
    ("dangerous", "harmless"), ("visible", "invisible"), ("permanent", "temporary"),
    ("official", "unofficial"), ("legal", "illegal"), ("public", "private"),
    ("professional", "amateur"), ("military", "civilian"), ("domestic", "foreign"),
    ("internal", "external"), ("manned", "unmanned"), ("armed", "unarmed"),
    ("active", "inactive"), ("active", "dormant"), ("simple", "complex"),
    ("easy", "difficult"), ("cheap", "expensive"), ("strong", "weak"),
    ("young", "old"), ("male", "female"), ("straight", "curved"),
    ("circular", "square"), ("solid", "hollow"), ("mandatory", "optional"),
    ("national", "regional"), ("annual", "monthly"), ("annual", "weekly"),
    ("annual", "daily"), ("monthly", "weekly"), ("monthly", "daily"),
    ("weekly", "daily"),
    # Shapes that turn up constantly in this corpus: settlements, species, railway
    # stations, retired footballers, defunct clubs.
    ("unincorporated", "incorporated"), ("defunct", "active"), ("disused", "active"),
    ("abandoned", "inhabited"), ("retired", "active"), ("medieval", "modern"),
    ("prehistoric", "modern"), ("contemporary", "ancient"), ("terrestrial", "aquatic"),
    ("freshwater", "marine"), ("nocturnal", "diurnal"), ("evergreen", "deciduous"),
    ("residential", "commercial"), ("tropical", "temperate"),
    ("landlocked", "coastal"), ("venomous", "harmless"), ("endangered", "widespread"),
]
ANTONYMS = {}
for _left, _right in ANTONYM_PAIRS:
    ANTONYMS.setdefault(_left, []).append(_right)
    ANTONYMS.setdefault(_right, []).append(_left)

# "one of the largest cities" supports "Is X large?" just as well as "a large city"
# does, and superlatives are far more common in opening sentences than bare adjectives.
# "latest"/"earliest"/"greatest"/"best" are deliberately absent - "the latest album"
# does not mean the album is late.
SUPERLATIVES = {
    "largest": "large", "biggest": "big", "smallest": "small", "tallest": "tall",
    "longest": "long", "shortest": "short", "widest": "wide", "narrowest": "narrow",
    "deepest": "deep", "shallowest": "shallow", "heaviest": "heavy",
    "lightest": "light", "densest": "dense", "highest": "high", "lowest": "low",
    "richest": "rich", "poorest": "poor", "rarest": "rare", "commonest": "common",
    "hottest": "hot", "coldest": "cold", "driest": "dry", "wettest": "wet",
    "fastest": "fast", "slowest": "slow", "strongest": "strong", "weakest": "weak",
    "oldest": "old", "newest": "new",
}

# "high in saturated fat" / "low in saturated fat" - the literal probe that failed.
COMPLEMENT_ADJ = {"high", "low", "rich", "poor", "abundant", "scarce"}

# Only a matter of degree can be hedged. "Is X generally large?" is a normal question;
# "Is X generally extinct?" is not, so the hedged templates are offered for these only.
GRADABLE = {
    "large", "small", "big", "tiny", "huge", "tall", "short", "long", "wide",
    "narrow", "deep", "shallow", "thick", "thin", "heavy", "light", "dense",
    "sparse", "high", "low", "rich", "poor", "abundant", "scarce", "hot", "cold",
    "warm", "cool", "wet", "dry", "humid", "arid", "hard", "soft", "smooth",
    "rough", "bright", "dim", "loud", "quiet", "fast", "slow", "ancient", "modern",
    "old", "new", "traditional", "common", "rare", "popular", "obscure", "famous",
    "dangerous", "harmless", "simple", "complex", "easy", "difficult", "cheap",
    "expensive", "strong", "weak", "young", "mountainous", "flat",
}
# A compound noun is not a property of the subject: a high school is not high, a heavy
# metal band is not heavy, a short film is not short. Capitalised compounds ("Cold War",
# "Northern Ireland") are already excluded by requiring a lowercase adjective.
BLOCKED_AFTER = {
    "high": {"school", "schools", "court", "street", "speed", "jump", "frequency",
             "definition", "tide", "seas", "energy", "society", "altitude", "voltage",
             "level", "military", "decoration", "rank", "award", "honour", "honor",
             "office", "authority", "governing", "priest"},
    "short": {"story", "stories", "film", "films", "track", "list", "wave", "circuit",
              "for"},
    "low": {"countries", "tide", "frequency", "earth", "level", "voltage", "budget"},
    "light": {"rail", "novel", "infantry", "cruiser", "industry", "aircraft",
              "machine", "bulb", "opera"},
    "heavy": {"metal", "industry", "water", "machinery", "cavalry", "cruiser"},
    "hard": {"rock", "drive", "disk", "water", "coal", "science", "core"},
    "soft": {"drink", "drinks", "rock", "power", "drug", "tissue", "ware"},
    "cold": {"war", "storage", "front", "blood", "case"},
    "hot": {"spring", "springs", "dog", "rod", "sauce", "air"},
    "short": {"story", "stories", "track", "list", "wave", "circuit", "version",
              "form", "name", "title"},
    "long": {"jump", "distance", "term", "range", "form", "island", "wave", "shot"},
    "fast": {"food", "bowling", "track", "forward"},
    "slow": {"motion", "cooker", "loris"},
    "open": {"source", "air", "water", "world", "access", "letter", "university"},
    "natural": {"gas", "history", "resources", "science", "sciences", "language",
                "selection", "number", "park", "reserve", "law", "philosophy"},
    "artificial": {"intelligence", "satellite", "neural", "insemination"},
    "common": {"law", "name", "names", "sense", "era", "room", "ground", "cold",
               "council", "stock"},
    "rare": {"earth", "book", "books", "disease"},
    "new": {"town", "wave", "year", "world", "testament", "deal", "age"},
    "old": {"town", "guard", "school", "world", "testament", "english", "city"},
    "public": {"school", "schools", "house", "transport", "domain", "health",
               "service", "sector", "library", "relations", "utility"},
    "private": {"school", "sector", "equity", "eye", "member", "practice", "law"},
    "national": {"park", "team", "guard", "anthem", "league", "museum", "forest",
                 "monument", "highway", "road", "register", "historic", "assembly",
                 "library", "trust", "service", "championship", "championships"},
    "regional": {"council", "district", "airport", "park", "league"},
    "solid": {"state", "waste", "rocket"},
    "male": {"voice", "line", "connector"},
    "female": {"voice", "line", "connector"},
    "military": {"academy", "junta", "police", "intelligence", "order"},
    "young": {"adult", "people", "man", "woman", "boy", "girl"},
    "strong": {"force", "interaction"},
    "simple": {"machine", "past", "harmonic", "majority"},
    "digital": {"camera", "audio", "signal", "media", "library"},
    "wild": {"boar", "west", "card", "rice", "type", "life"},
    "real": {"estate", "number", "numbers", "time", "madrid", "tennis"},
    "straight": {"line", "flush", "edge"},
    "circular": {"saw", "economy", "motion"},
    "annual": {"report", "meeting", "average", "rainfall", "plant", "herb",
               "herbaceous", "grass", "flowering", "species"},
    "daily": {"mail", "telegraph", "express", "record", "mirror", "life"},
    "commercial": {"bank", "aviation", "vehicle", "break", "radio", "airline",
                   "fishing", "district", "fm"},
    "terrestrial": {"television", "broadcasting"},
    "marine": {"corps", "biology", "life", "park", "reserve"},
    "medieval": {"latin", "greek", "warm"},
    "contemporary": {"art", "music", "dance", "history"},
    "active": {"volcano", "service", "duty", "site", "matrix", "ingredient"},
    "flat": {"track", "cap", "earth", "screen", "racing", "race"},
    "retired": {"list"},
    "endangered": {"species"},
    "external": {"link", "links"},
    "internal": {"combustion", "revenue", "medicine", "affairs"},
    "domestic": {"violence", "cat", "animal", "flight", "flights", "policy", "market"},
    "foreign": {"affairs", "minister", "policy", "exchange", "language", "legion",
                "relations", "secretary", "service"},
    "modern": {"art", "pentathlon", "english", "history", "greek"},
    "traditional": {"chinese", "medicine", "music", "dance", "arts"},
    "professional": {"wrestling", "development", "services", "association"},
    "amateur": {"radio", "athletics", "boxing", "dramatics"},
    "popular": {"culture", "music", "front", "vote", "science"},
    "urban": {"district", "planning", "legend", "area"},
    "rural": {"district", "development", "council"},
    "big": {"band", "bang", "brother", "data", "cat", "ten", "east"},
    "small": {"arms", "business", "intestine", "talk", "press", "claims"},
    "large": {"intestine", "hadron", "scale"},
    "deep": {"south", "sea", "space", "learning", "web", "water"},
    "wide": {"receiver", "gauge", "area", "web", "screen"},
    "narrow": {"gauge", "band", "escape"},
    "warm": {"front", "blooded", "up"},
    "cool": {"jazz", "down"},
    "dry": {"cleaning", "dock", "county", "season", "ice", "stone"},
    "wet": {"season", "nurse", "land", "suit"},
    "legal": {"system", "aid", "person", "code", "tender", "profession"},
    "official": {"language", "languages", "residence", "website", "gazette"},
    "permanent": {"secretary", "representative", "member", "way", "deacon"},
    "temporary": {"exhibition", "housing", "worker"},
    "dangerous": {"goods"},
    "visible": {"light", "spectrum"},
    "eastern": {"orthodox", "bloc", "front", "time", "conference"},
    "western": {"front", "bloc", "time", "conference", "australia", "sahara"},
    "northern": {"lights", "hemisphere", "territory", "front", "district"},
    "southern": {"hemisphere", "cross", "ocean", "district", "conference"},
    "central": {"bank", "nervous", "processing", "station", "committee", "district"},
    "manned": {"mission", "flight"},
    "extinct": {"volcano"},
}

# Words that stop an adjective from describing the subject: "not large", "less common",
# "one of the more common". Superlatives arrive as their own token, so "most" is safe
# to block outright.
BLOCK_BEFORE = {"not", "no", "less", "least", "more", "most", "than", "nor", "never",
                "rarely", "occasionally", "sometimes", "semi", "part", "half", "non"}
ALLOWED_BEFORE = {"a", "an", "the", "very", "quite", "relatively", "fairly",
                  "extremely", "highly", "particularly", "especially", "unusually",
                  "generally", "typically", "mostly", "mainly", "largely", "chiefly",
                  "predominantly", "primarily"}
# An adjective is followed by the noun it describes. "a British official in India" is a
# noun; "Water privatization is short for ..." is an idiom. Both end here.
FOLLOW_STOP = {"in", "of", "on", "at", "for", "from", "to", "with", "by", "and", "or",
               "that", "which", "who", "as", "near", "between", "within", "during",
               "under", "over", "after", "before", "is", "was", "are", "were", "has",
               "had", "but", "than", "when", "where", "into", "about", "against"}
# "a species of small snail" still describes the subject. "a member of the small
# parliament" would not, so only type-of nouns open that door.
KIND_WORDS = {"species", "genus", "subspecies", "subgenus", "family", "subfamily",
              "type", "kind", "variety", "breed"}
# A complement noun must be bare ("in saturated fat", "in iron"). "high in the mountains"
# is a location, not a property.
COMPLEMENT_STOP = {"the", "a", "an", "of", "and", "or", "which", "that", "with", "for",
                   "from", "to", "by", "on", "at", "in", "its", "their", "this", "these"}

# Every template is written lowercase and sentence-cased on the way out, so the same
# string can serve as "Was the ..." and as "was the ...". {said} keeps the article the
# article used ("a Tesla coil"); {said_the} is the object form ("the Tesla coil"). The
# third column is a selection weight: active voice is weighted up because
# "Did Nikola Tesla invent the Tesla coil?" measured P(Yes)=0.0000 - the model did not
# recognise it as a yes/no question at all.
CREATOR_TEMPLATES = [
    ("{bepast} {said} {verb} by {name}?", (), 1),
    ("{bepast} {said_the} {verb} by {name}?", (), 1),
    ("{bepast} {said_the} originally {verb} by {name}?", (), 1),
    ("{bepast} {said} really {verb} by {name}?", (), 1),
    ("{has} {said_the} been {verb} by {name}?", (), 1),
    ("is it true that {said} {bepast} {verb} by {name}?", (), 1),
    ("is it correct that {said_the} {bepast} {verb} by {name}?", (), 1),
    ("{said_the} {bepast} {verb} by {name}, right?", (), 1),
    ("{bepast} {said_the} the work of {name}?", (), 1),
    ("did {name} {base} {said_the}?", ("base",), 3),
    ("did {name} really {base} {said_the}?", ("base",), 3),
    ("was it {name} who {past} {said_the}?", ("past",), 1),
    ("is {name} the one who {past} {said_the}?", ("past",), 1),
    ("is it right that {name} {past} {said_the}?", ("past",), 1),
    ("{name} {past} {said_the}, right?", ("past",), 1),
]
YEAR_TEMPLATES = [
    ("{bepast} {said} {verb} in {year}?", (), 1),
    ("{bepast} {said_the} {verb} in {year}?", (), 1),
    ("{bepast} {said} {verb} in the year {year}?", (), 1),
    ("{bepast} {said_the} {verb} during {year}?", (), 1),
    ("is it true that {said} {bepast} {verb} in {year}?", (), 1),
    ("is it correct that {said_the} {bepast} {verb} in {year}?", (), 1),
    ("was it in {year} that {said} {bepast} {verb}?", (), 1),
    ("was {year} the year {said_the} {bepast} {verb}?", (), 1),
    ("{said_the} {bepast} {verb} in {year}, right?", (), 1),
    ("{do} {said_the} date from {year}?", (), 1),
    ("did {said_the} {base} in {year}?", ("base",), 3),
    ("did {said_the} first {base} in {year}?", ("base",), 3),
]
HEAD_TEMPLATES = [
    ("{cop} {said} {head}?", (), 1),
    ("{cop} {said_the} {head}?", (), 1),
    ("{cop} {said} really {head}?", (), 1),
    ("{cop} {said} considered {head}?", (), 1),
    ("is it true that {said} {cop} {head}?", (), 1),
    ("is it correct to say that {said} {cop} {head}?", (), 1),
    ("would you say {said} {cop} {head}?", (), 1),
    ("would it be right to call {said_the} {head}?", (), 1),
    ("can {said} be described as {head}?", (), 1),
    ("can {said_the} be called {head}?", (), 1),
    ("{do} {said_the} count as {head}?", (), 1),
    ("{has} {said} been described as {head}?", (), 1),
    ("{said_the} {cop} {head}, right?", (), 1),
]
ADJECTIVE_TEMPLATES = [
    ("{cop} {said} {adj}?", (), 2),
    ("{cop} {said_the} {adj}?", (), 2),
    ("{cop} {said} generally {adj}?", ("gradable",), 1),
    ("{cop} {said_the} usually {adj}?", ("gradable",), 1),
    ("{cop} {said} considered {adj}?", (), 1),
    ("{cop} {said} really {adj}?", (), 1),
    ("{cop} {said} pretty {adj}?", ("gradable",), 1),
    ("{cop} {said_the} known for being {adj}?", ("gradable",), 1),
    ("is it true that {said} {cop} {adj}?", (), 1),
    ("would you say {said} {cop} {adj}?", (), 1),
    ("would you describe {said_the} as {adj}?", (), 1),
    ("can {said} be described as {adj}?", (), 1),
    ("{said_the} {cop} {adj}, right?", (), 1),
]
TEMPLATES = {"creator": CREATOR_TEMPLATES, "year": YEAR_TEMPLATES,
             "head": HEAD_TEMPLATES, "adjective": ADJECTIVE_TEMPLATES}

FIELD = re.compile(r"\{(\w+)\}")


def speak(subject):
    """'A Tesla coil' reads wrong mid-sentence; 'a Tesla coil' does not."""
    first, _, rest = subject.partition(" ")
    if first in ("A", "An", "The") and rest:
        return f"{first.lower()} {rest}"
    return subject


def as_object(subject):
    """Object position wants the definite article: 'Did Nikola Tesla invent the Tesla coil?'"""
    first, _, rest = subject.partition(" ")
    if first.lower() in ("a", "an", "the") and rest:
        return f"the {rest}"
    return subject


def fill(template, slots):
    """str.format would explode on a stray '{' inside a Wikipedia phrase."""
    return FIELD.sub(lambda m: slots[m.group(1)], template)


def indefinite(text):
    return text.split(" ", 1)[0].lower() in ("a", "an")


def printable(text):
    """Console output only - this file is generated on a cp1252 terminal."""
    return text.encode("ascii", "replace").decode("ascii")


def restyle(question, style):
    """Real users skip the capital and the question mark; the label must not care."""
    if style in ("lower_all", "lower_all_no_mark"):
        question = question.lower()
    elif style == "lower_start":
        question = question[0].lower() + question[1:]
    if style in ("no_mark", "lower_all_no_mark"):
        question = question.rstrip("?").rstrip()
    return question


def render_pair(kind, slots, true_slots, false_slots, style, rng):
    """One template, one style, both halves - anything else teaches phrasing = label."""
    usable = [(text, weight) for text, requires, weight in TEMPLATES[kind]
              if all(true_slots.get(key) for key in requires)]
    template = rng.choices([text for text, _ in usable],
                           weights=[weight for _, weight in usable])[0]
    questions = []
    for extra in (true_slots, false_slots):
        merged = dict(slots)
        merged.update(extra)
        text = fill(template, merged)
        questions.append(restyle(text[0].upper() + text[1:], style))
    if questions[0] == questions[1]:
        return None
    return template, [(questions[0], "Yes"), (questions[1], "No")]


def predicate_head(predicate):
    # Drop the parenthetical rather than cutting at it: "one of the 288 Vidhan Sabha
    # (assembly) constituencies" must not become "one of the 288 Vidhan Sabha".
    plain = " ".join(PARENTHETICAL.sub(" ", predicate).split())
    cut = re.split(r"[,;]", plain)[0].strip().rstrip(".")
    words = cut.split()
    if not 2 <= len(words) <= 9 or words[0].lower() not in HEAD_START:
        return None
    if HEAD_REJECT.match(cut):
        return None
    if "(" in cut or ")" in cut:
        return None
    if cut != plain.strip().rstrip("."):
        # Cut at a comma. "a group of companies that have supplier, distributor and ..."
        # would survive as "a group of companies that have supplier", and "the commercial,
        # cultural and educational centre" as "the commercial" - grammatical but beheaded.
        if len(words) < 3 or RELATIVE.search(cut) or cut.rsplit(" ", 1)[-1].lower() in DANGLING:
            return None
    if INITIAL_END.search(cut):
        return None
    return cut


def complement_of(tokens, start):
    """'high in saturated fat' -> 'saturated fat'; 'high in the mountains' -> nothing."""
    words = []
    for token in tokens[start:start + 3]:
        word = token.strip(".,;:'\"").lower()
        if not word.isalpha() or word in COMPLEMENT_STOP:
            break
        words.append(word)
    return " ".join(words) if words else None


def find_adjective(predicate, sentence, subject):
    """Mine one gradable property that genuinely describes the subject.

    Deliberately strict: the adjective must sit at the front of the predicate's first
    clause, modifying its head noun, with nothing but determiners, degree words and
    proper modifiers in front of it. "a port serving the inland town of Astros" is
    about the town, not the port, and that is the failure mode this rules out.
    """
    clause = re.split(r"[,;(]", predicate)[0].strip().rstrip(".")
    tokens = clause.split()
    subject_words = {w.lower() for w in re.findall(r"[A-Za-z]+", subject)}
    for i, token in enumerate(tokens):
        if token != token.lower():          # capitalised -> part of a proper name
            continue
        word = token.strip(".,;:'\"")
        if not word.isalpha():
            continue
        adjective = SUPERLATIVES.get(word, word)
        if adjective not in ANTONYMS or adjective in subject_words:
            continue
        raw_before = [t.strip(".,;:'\"") for t in tokens[:i]]
        before = [t.lower() for t in raw_before]
        if any(b in BLOCK_BEFORE for b in before):
            continue

        start = 0
        if "of" in before:
            pivot = before.index("of")
            # "a species of small snail" keeps describing the subject, and "one of the
            # largest cities" is the superlative shape that survives; nothing else does.
            if pivot and before[pivot - 1] in KIND_WORDS:
                start = pivot + 1
            elif before == ["one", "of", "the"] and word in SUPERLATIVES:
                start = i
            else:
                continue
        complement = None
        if word in COMPLEMENT_ADJ and i + 1 < len(tokens) and tokens[i + 1].lower() == "in":
            complement = complement_of(tokens, i + 2)

        if complement is not None:
            # "a fruit high in saturated fat" post-modifies the noun in front of it, so
            # the determiner-only rule below cannot apply - the head noun is supposed to
            # be there. Allow it and its own modifiers, nothing longer.
            content = sum(1 for j in range(start, i) if before[j] not in ALLOWED_BEFORE)
            if i > 5 or content > 3:
                continue
        else:
            if any(before[j] not in ALLOWED_BEFORE and not raw_before[j][:1].isupper()
                   for j in range(start, i)):
                continue
            if i > 5 or i + 1 >= len(tokens):
                continue
            following = tokens[i + 1].strip(".,;:'\"").lower()
            if not following.replace("-", "").isalpha() or following in FOLLOW_STOP:
                continue
            blocked = BLOCKED_AFTER.get(adjective, ())
            if following in blocked or following.split("-")[0] in blocked:
                continue

        # An antonym already in the sentence means the false half might be true too.
        options = [a for a in ANTONYMS[adjective]
                   if not re.search(rf"\b{a}\b", sentence, re.I) and a not in subject_words]
        if not options:
            continue
        gradable = adjective in GRADABLE
        if complement:
            return (f"{adjective} in {complement}",
                    [f"{a} in {complement}" for a in options], gradable)
        return adjective, options, gradable
    return None


def parse(sentence):
    """Pull the subject and whichever facts the opening sentence actually states."""
    if RUN_ON.search(sentence) or DISAMBIGUATION.search(sentence):
        return None
    match = DEFINITION.match(sentence + " ")
    if not match:
        return None
    subject = PARENTHETICAL.sub("", match.group(1).strip()).strip(" ,;:")
    if not (3 <= len(subject) <= 60) or "(" in subject or ")" in subject:
        return None
    if "," in subject or REJECT.match(subject) or len(subject.split()) > 8:
        return None
    if subject.lower() in PRONOUN_SUBJECTS:
        return None
    if YEAR.search(subject):
        return None

    predicate = match.group(2).strip()
    # An initial is allowed to contain a full stop, so RUN_ON cannot see a weld that
    # happens to land on one ("... and S. Slastyonka () is a rural locality ..."). If
    # what follows the definition is itself a definition, two articles are stuck together.
    leftover = (sentence + " ")[match.end(0):]
    if len(leftover) > 25 and DEFINITION.match(leftover):
        return None
    # The sentence's own copula settles both number and tense, so questions agree with
    # the subject: "are coconuts ...", "was the theatre ...".
    after = (sentence + " ")[match.end(1):].lstrip()
    copula = after.split(" ", 1)[0].lower()
    if copula not in ("is", "was", "are", "were"):
        return None

    facts = {"copula": copula, "head": predicate_head(predicate)}

    # Search the definition itself, never the whole line: if two articles have been
    # welded together the tail belongs to a different subject entirely.
    stated = match.group(0)
    person = bool(PERSON_HINT.search(stated[:match.end(1) + 40]))
    creator = CREATOR_VERB.search(stated)
    if creator and not person:
        words = [w for w in creator.group(2).split() if w not in DESCRIPTORS]
        if words:
            # "... directed by Santosh Sivan." captures the full stop, which then reads
            # "Was Mumbaikar directed by Santosh Sivan.?". An initial ("J. R. R.") keeps
            # its dot; an ordinary surname does not.
            name = re.sub(r"(?<=\w\w)\.$", "", " ".join(words))
            tail = stated[creator.end(2):].split()
            truncated = bool(tail) and tail[0].lower() in PARTICLES
            if not INITIAL_END.search(name) and not truncated:
                facts["creator"] = (creator.group(1), name)

    dated = DATED_VERB.search(stated)
    if dated and not person:
        facts["year"] = (dated.group(1), dated.group(2))

    adjective = find_adjective(predicate, sentence, subject)
    if adjective:
        facts["adjective"] = adjective
    return subject, facts


def make_pair(subject, facts, pools, rivals, context, rng):
    """One true question and one false one, same shape, so only the fact differs.

    The false value is taken from another passage in the SAME context whenever one
    is available: then the wrong answer is sitting right there in the text, so it
    cannot be dismissed as implausible - the relation has to be read.
    """
    kinds = [k for k in ("creator", "year", "head", "adjective") if facts.get(k)]
    if not kinds:
        return None
    kind = rng.choices(kinds, weights=[KIND_WEIGHTS[k] for k in kinds])[0]
    plural = facts["copula"] in ("are", "were")
    slots = {
        "said": speak(subject),
        "said_the": as_object(subject),
        "cop": facts["copula"],
        "be": "are" if plural else "is",
        "bepast": "were" if plural else "was",
        "do": "do" if plural else "does",
        "has": "have" if plural else "has",
    }
    style = rng.choices(list(STYLE_WEIGHTS), weights=list(STYLE_WEIGHTS.values()))[0]

    def in_context(candidate):
        """Is this value physically present in the passage the reader is given?

        This is the whole anti-shortcut rule. If the wrong answer never appears in the context,
        then "the name in the question is not in the text" identifies the false half on its own,
        and the model can score without reading the relation. Measured on the first build: word
        overlap alone separated Yes from No with AUC 0.740 and 71.6% accuracy, against 0.560 and
        54.8% for human-written BoolQ. Requiring presence removes the signal at the source.
        """
        head = str(candidate).split()[0]
        return bool(re.search(rf"\b{re.escape(head)}\b", context, re.I))

    def wrong(key, true_value, extra_ok):
        options = list(rivals.get(key) or [])
        rng.shuffle(options)
        if pools[key]:
            options += [rng.choice(pools[key]) for _ in range(20)]
        for candidate in options:
            if candidate != true_value and extra_ok(candidate) and in_context(candidate):
                return candidate
        return None

    if kind == "creator":
        verb, name = facts["creator"]
        other = wrong("creators", name,
                      lambda c: c not in name and name not in c)
        if not other:
            return None
        base, past = ACTIVE_FORMS.get(verb, (None, None))
        shared = {"verb": verb, "base": base, "past": past}
        return kind, render_pair(kind, slots,
                                 dict(shared, name=name), dict(shared, name=other),
                                 style, rng)

    if kind == "year":
        verb, year = facts["year"]
        # Constrain the rival to a nearby year. Drawing freely from the global pool produced
        # things like "Was <a 2019 film> released in 1523?", which is answerable from era
        # alone - the model could score it without reading the passage at all.
        other = wrong("years", year,
                      lambda c: abs(int(c) - int(year)) <= RIVAL_YEAR_SPREAD)
        if not other:
            # Previously this invented a nearby year with rng.randint. That year was, by
            # construction, absent from the context - which is exactly the tell being removed
            # above. Dropping the pair costs volume; inventing one costs correctness.
            return None
        if other == year:
            return None
        shared = {"verb": verb, "base": INTRANSITIVE_FORMS.get(verb)}
        return kind, render_pair(kind, slots,
                                 dict(shared, year=year), dict(shared, year=other),
                                 style, rng)

    if kind == "adjective":
        adjective, options, gradable = facts["adjective"]
        # The antonym must be physically present in the context, attached to some other
        # subject - the same rule the creator/year rivals follow. The old fallback to
        # options[0] emitted an antonym that was nowhere in the passage, which is the
        # lexical tell this build exists to remove, so it now drops the pair instead.
        other = next((o for o in options
                      if re.search(rf"\b{o.split()[0]}\b", context, re.I)), None)
        if not other:
            return None
        mark = {"gradable": "yes" if gradable else ""}
        return kind, render_pair(kind, slots,
                                 dict(mark, adj=adjective), dict(mark, adj=other),
                                 style, rng)

    head = facts["head"]
    true_words = content_words(head)
    # No shared words means the false claim cannot accidentally be true as well. The
    # false head also has to open the same way: if "a ..." only ever showed up on the
    # No side of plural subjects, the article alone would give the answer away.
    other = wrong("heads", head,
                  lambda c: not (content_words(c) & true_words)
                  and indefinite(c) == indefinite(head))
    if not other:
        return None
    return kind, render_pair(kind, slots, {"head": head}, {"head": other}, style, rng)


def build_context(source, subject, sentences, by_word, rng, themed=()):
    wanted = rng.choices(list(DISTRACTOR_WEIGHTS), weights=list(DISTRACTOR_WEIGHTS.values()))[0]
    near = set()
    for word in content_words(subject):
        near.update(by_word.get(word, ()))
    near = tuple(sorted(near))   # sorted, so a set's iteration order cannot leak in

    chosen, seen = [], {source}
    for _ in range(wanted * 12):
        if len(chosen) == wanted:
            break
        # A passage carrying the opposite adjective puts the wrong answer physically in
        # the text, attached to someone else - the same job the rival creators do.
        if themed and rng.random() < THEMED_CHANCE:
            candidate = sentences[rng.choice(themed)]
        elif near and rng.random() < NEAR_MISS_CHANCE:
            candidate = sentences[rng.choice(near)]
        else:
            candidate = sentences[rng.randrange(len(sentences))]
        # A passage that also describes the subject could contradict the label.
        if candidate in seen or subject.lower() in candidate.lower():
            continue
        seen.add(candidate)
        chosen.append(candidate)

    chosen.insert(rng.randint(0, len(chosen)), source)
    return " ".join(chosen), [s for s in chosen if s != source]


def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    parsed, creators, heads, years = [], [], [], []
    head_only = 0

    for count, article in enumerate(iter_articles(str(SOURCE))):
        if count >= MAX_ARTICLES or len(parsed) >= TARGET_PAIRS:
            break
        sentence = first_sentence(article)
        if not sentence:
            continue
        result = parse(sentence)
        if not result:
            continue
        subject, facts = result
        if not any(facts.get(k) for k in ("creator", "year", "head", "adjective")):
            continue
        # Sentences stating only a description are the overwhelming majority; take them
        # until the quota is met and keep reading for the rarer, more useful kinds.
        if not any(facts.get(k) for k in ("creator", "year", "adjective")):
            if head_only >= HEAD_ONLY_SHARE * TARGET_PAIRS:
                continue
            head_only += 1
        parsed.append((subject, facts, sentence))
        if facts.get("creator"):
            creators.append(facts["creator"][1])
        if facts.get("head"):
            heads.append(facts["head"])
        if facts.get("year"):
            years.append(facts["year"][1])
        if count % 50_000 == 0 and count:
            print(f"  {count:,} articles read -> {len(parsed):,} candidate pairs")

    pools = {"creators": creators, "heads": heads, "years": years}
    print(f"  pools: {len(creators):,} real creators, {len(heads):,} real descriptions, "
          f"{len(years):,} real years")

    facts_of = {sentence: facts for _, facts, sentence in parsed}
    sentences = sorted(facts_of)
    by_word = {}
    for i, sentence in enumerate(sentences):
        for word in content_words(sentence[:80]):
            by_word.setdefault(word, []).append(i)

    rows = []
    grounded = 0
    kind_counts, template_counts, shown = {}, {}, {}
    antonym_in_context = adjective_rows = 0
    for subject, facts, sentence in parsed:
        themed = ()
        if facts.get("adjective"):
            pool = []
            for option in facts["adjective"][1]:
                pool += by_word.get(option.split()[0], ())
            themed = tuple(pool)
        context, distractors = build_context(sentence, subject, sentences, by_word,
                                             rng, themed)
        rivals = {"creators": [], "heads": [], "years": []}
        for other in distractors:
            other_facts = facts_of.get(other, {})
            if other_facts.get("creator"):
                rivals["creators"].append(other_facts["creator"][1])
            if other_facts.get("year"):
                rivals["years"].append(other_facts["year"][1])
            if other_facts.get("head"):
                rivals["heads"].append(other_facts["head"])

        made = make_pair(subject, facts, pools, rivals, context, rng)
        if not made or not made[1]:
            continue
        kind, (template, pair) = made
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        template_counts[template] = template_counts.get(template, 0) + 1
        if len(shown.setdefault(kind, [])) < 2:
            shown[kind].append((pair, context))
        if kind == "adjective":
            adjective_rows += 1
            opposites = {o.split()[0] for o in facts["adjective"][1]}
            if any(re.search(rf"\b{word}\b", other, re.I)
                   for other in distractors for word in opposites):
                antonym_in_context += 1
        if any(v for v in rivals.values()):
            grounded += 1
        # Both halves share one context - that is what makes the contrast bite.
        rows.append([(question, context, answer) for question, answer in pair])

    print(f"  {grounded:,} pairs had a rival fact available inside their own context")
    if adjective_rows:
        print(f"  {antonym_in_context:,} of {adjective_rows:,} adjective pairs had the "
              f"opposite adjective present in a distractor passage")
    total = sum(kind_counts.values()) or 1
    print("  fact types: " + ", ".join(
        f"{kind} {count:,} ({count / total:.0%})" for kind, count in sorted(kind_counts.items())))
    print(f"  templates used: {len(template_counts)} of "
          f"{sum(len(v) for v in TEMPLATES.values())}")
    # Standing project rule: eyeball generated data before training on it.
    for kind in sorted(shown):
        print(f"  --- {kind} ---")
        for pair, context in shown[kind]:
            for question, answer in pair:
                print(f"      {answer:3} | {printable(question)}")
            print(f"          context: {printable(context[:100])}...")

    # Split by pair, not by row: a half-pair in validation would be scored against a
    # context whose other half was trained on, and each file would drift off 50/50.
    rng.shuffle(rows)
    split = int(len(rows) * (1 - VALIDATION_FRACTION))
    for name, pairs in (("train", rows[:split]), ("validation", rows[split:])):
        # Decide per PAIR, before flattening, so both halves of a contrast are treated alike.
        # Stripping only one half would leave the two questions incomparable.
        subset = []
        bare_pairs = 0
        for pair in pairs:
            bare = rng.random() < NO_CONTEXT_FRACTION
            bare_pairs += bare
            for question, context, answer in pair:
                subset.append((question, None if bare else context, answer))
        rng.shuffle(subset)
        path = output_dir / f"{name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for question, context, answer in subset:
                if context:
                    f.write(f"User: {question}\n\nContext: {context}\n"
                            f"Assistant: {answer}\n<|endoftext|>\n")
                else:
                    f.write(f"User: {question}\n"
                            f"Assistant: {answer}\n<|endoftext|>\n")
        yes = sum(1 for r in subset if r[2] == "Yes")
        print(f"Done: {path} - {len(subset):,} examples, {yes:,} yes / {len(subset) - yes:,} no"
              f" | {bare_pairs:,} of {len(pairs):,} pairs written with no context "
              f"({bare_pairs / max(len(pairs), 1):.0%})")


if __name__ == "__main__":
    main()
