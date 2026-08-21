"""Stress-test multi-turn conversation and instruction-following (chat mode).

Scope: persona stability, coreference, contradiction tracking, format constraints and
degradation with conversation length. Question answering is covered by tools/stress_qa.py
and is deliberately not repeated here.

Method rules this file obeys, because the project has been burned by each of them:

  N >= 5 everywhere.  Every number below is a rate over repeats, never one transcript.

  Segment before averaging.  Populations that are NOT pooled: persona neutral vs adversarial,
  coreference terse vs chatty register, format constraint type, chat vs auto mode, early vs
  late conversation position, SODA vs PersonaChat replay.  Each is reported on its own line.

  Test the shape it was trained on.  Suite 6 replays REAL held-out dialogues from
  data/soda/validation.txt and data/personachat/validation.txt through the same pipeline:
  true prior turns as history, the true next user turn as the question.  That is the
  in-distribution baseline every hand-written probe is compared against, so a probe written
  in the wrong register cannot masquerade as a broken model (onboarding section 9).

  Measure the PRODUCT, not the model.  History is handled exactly as chat.py does it:
  an empty answer is not appended, and history is capped at MAX_HISTORY_TURNS = 8.  The cap
  is itself a measured cause of long-range memory failure, so the harness records whether the
  fact was still in the prompt when the question was asked (fact_in_prompt) - that separates
  "the model forgot" from "the serving layer dropped it".

  Attribute, do not blame.  Route category, confidence, retrieval use, passage count and
  dropped-candidate count are logged for every single generation.

Usage:
    .\\.venv\\Scripts\\python.exe -m tools.stress_conversation --repeats 5
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from orchestrator import config
from orchestrator.pipeline import Assistant

OUT = Path("tests/stress_conversation.txt")

# chat.py keeps at most this many turns and drops empty answers. Copied, not imported, so the
# harness measures the served behaviour even if someone edits chat.py.
MAX_HISTORY_TURNS = 8

SODA_TRAIN = Path("data/soda/train.txt")
PERSONACHAT_TRAIN = Path("data/personachat/train.txt")
NO_ROBOTS_TRAIN = Path("data/no_robots/train.txt")
OASST1_TRAIN = Path("data/oasst1/train.txt")

SODA_VAL = Path("data/soda/validation.txt")
PERSONACHAT_VAL = Path("data/personachat/validation.txt")

VERDICT_WORDS = {"yes", "no", "true", "false", "unknown"}

# Words that follow "I'm ..." without being a name. Without this list the extractor reports
# "I'm Sorry" and "I'm Not" as identities and every conversation looks unstable.
NOT_NAMES = {
    "sorry", "not", "just", "here", "glad", "good", "fine", "great", "happy", "sure",
    "really", "afraid", "doing", "going", "still", "the", "a", "an", "it", "i", "you",
    "we", "they", "he", "she", "my", "your", "no", "yes", "ok", "okay", "well", "also",
    "always", "never", "very", "so", "too", "now", "then", "thanks", "thank", "hi",
    "hello", "hey", "nice", "trying", "looking", "working", "thinking", "sorryi",
    "true", "false", "unknown", "oh", "what", "that", "this", "there", "all", "one",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "god", "yeah", "right", "cool",
    "and", "but", "or", "if", "when", "who", "how", "why", "am", "are", "is", "was",
}

COLOURS = {
    "red", "blue", "green", "yellow", "orange", "purple", "violet", "black", "white",
    "pink", "brown", "grey", "gray", "gold", "silver", "beige", "teal", "cyan", "magenta",
    "indigo", "maroon", "turquoise", "lavender", "crimson", "scarlet",
}

FRUITS = {
    "apple", "banana", "orange", "pear", "grape", "grapes", "mango", "peach", "plum",
    "cherry", "cherries", "strawberry", "strawberries", "lemon", "lime", "kiwi",
    "pineapple", "melon", "watermelon", "apricot", "fig", "papaya", "blueberry",
    "raspberry", "coconut", "avocado", "tomato", "date", "guava", "nectarine",
}

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


# --------------------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------------------

def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text or "")


def word_count(text: str) -> int:
    return len(words(text))


def sentence_count(text: str) -> int:
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return len([s for s in re.split(r"[.!?]+", stripped) if s.strip()]) or 1


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def is_verdict_only(text: str) -> bool:
    w = words(text)
    return len(w) == 1 and w[0].lower() in VERDICT_WORDS


def opens_with_verdict(text: str) -> bool:
    w = words(text)
    return bool(w) and w[0].lower() in VERDICT_WORDS


# ⚠ These were case-SENSITIVE in the first draft of this file, which silently missed
# "Hello, I'm Max." and "It's Kevin." and scored a conversation that used four different
# names as zero. Measured before/after on the same transcripts: 0.2 -> 2.4 mean distinct
# names per adversarial conversation. re.IGNORECASE on the LEAD phrase only; the captured
# name must still be capitalised, which is what keeps "I'm sorry" out.
_NAME_LEAD_STRICT = re.compile(
    r"\b(?:my name(?:'s| is)|i am|i'm|call me|they call me|you can call me)\s+"
    r"((?:[A-Z][A-Za-z'\-]{1,19})(?:\s+[A-Z][A-Za-z'\-]{1,19})?)",
    re.IGNORECASE)
# "It's Kevin." is how this model most often answers "what's your name?", so a strict-only
# extractor understates the instability. Reported as a separate number, never merged.
_NAME_LEAD_LOOSE = re.compile(
    r"\b(?:it(?:'s| is)|this is|they call me)\s+((?:[A-Z][A-Za-z'\-]{1,19}))",
    re.IGNORECASE)
_NAME_BARE = re.compile(r"^\s*((?:[A-Z][A-Za-z'\-]{1,19}))\s*[.!]?\s*$")


def _clean_names(hits) -> list:
    """Trim a captured span down to its leading run of real name tokens.

    ⚠ re.IGNORECASE applies to the WHOLE pattern, so [A-Z] inside these regexes also matches
    lowercase and the two-word capture greedily swallows "Sarah and". Trimming rather than
    rejecting is what keeps "My name is Sarah and ..." scoring as ONE name: the first draft
    rejected the whole match and reported zero names, the second counted "Sarah" and
    "Sarah and" as two different identities. Both would have misreported persona stability.
    """
    found = []
    for hit in hits:
        parts = []
        for piece in hit.split():
            if len(piece) < 2 or not piece[:1].isupper() or piece.lower() in NOT_NAMES:
                break
            parts.append(piece)
        if parts:
            found.append(" ".join(parts))
    return found


def extract_self_names(answer: str) -> list:
    """Names the assistant explicitly claims for ITSELF ("I'm X", "my name is X")."""
    text = answer or ""
    return _clean_names(_NAME_LEAD_STRICT.findall(text)) + _clean_names(_NAME_BARE.findall(text))


def extract_self_names_loose(answer: str) -> list:
    """Adds the "It's X" form. Only meaningful when the question was about the name."""
    return sorted(set(extract_self_names(answer))
                  | set(_clean_names(_NAME_LEAD_LOOSE.findall(answer or ""))))


def three_colour_check(answer: str) -> tuple[bool, bool]:
    """(strict, loose). Strict = three distinct colours and nothing else."""
    low = [w.lower() for w in words(answer)]
    mentioned = [w for w in low if w in COLOURS]
    distinct = set(mentioned)
    loose = len(distinct) >= 3
    filler = {"and", "the", "a"}
    extra = [w for w in low if w not in COLOURS and w not in filler]
    strict = len(distinct) == 3 and len(mentioned) == 3 and not extra
    return strict, loose


def bare_number(answer: str) -> tuple[bool, int | None]:
    stripped = (answer or "").strip().rstrip(".!").strip()
    if re.fullmatch(r"\d{1,4}", stripped):
        return True, int(stripped)
    if stripped.lower() in NUMBER_WORDS:
        return True, NUMBER_WORDS[stripped.lower()]
    return False, None


# --------------------------------------------------------------------------------------
# corpus audit - what the model was actually trained on
# --------------------------------------------------------------------------------------

def audit_dialogue_corpus(path: Path, max_lines: int | None = None) -> dict:
    """Assistant-turn statistics for a `User:/Assistant:` corpus, per dataset (never pooled)."""
    if not path.exists():
        return {"path": str(path), "present": False}

    self_names: Counter = Counter()
    first_tokens: Counter = Counter()
    assistant_turns = 0
    dialogues = 0
    dialogues_with_self_name = 0
    named_here = False
    user_addresses_assistant_by_name = 0
    answer_lengths: list[int] = []

    intro = re.compile(r"\b(?:my name is|my name's|i am|i'm)\s+([A-Z][A-Za-z'\-]{1,19})\b",
                       re.IGNORECASE)
    addressed = re.compile(r"^User:\s*(?:Hi|Hey|Hello|Well|So|Okay|Oh)?[,\s]*([A-Z][a-z]{2,15}),")

    with path.open(encoding="utf-8", errors="replace") as handle:
        for index, raw in enumerate(handle):
            if max_lines is not None and index >= max_lines:
                break
            line = raw.rstrip("\n")
            if line.strip() == "<|endoftext|>":
                dialogues += 1
                dialogues_with_self_name += int(named_here)
                named_here = False
                continue
            if line.startswith("Assistant:"):
                body = line[len("Assistant:"):].strip()
                assistant_turns += 1
                answer_lengths.append(word_count(body))
                token = (words(body) or [""])[0].lower()
                first_tokens[token] += 1
                for hit in intro.findall(body):
                    if hit.lower() not in NOT_NAMES and hit[:1].isupper():
                        self_names[hit] += 1
                        named_here = True
            elif line.startswith("User:"):
                if addressed.match(line):
                    user_addresses_assistant_by_name += 1

    total_first = sum(first_tokens.values()) or 1
    verdict_share = sum(first_tokens[w] for w in VERDICT_WORDS) / total_first
    return {
        "path": str(path),
        "present": True,
        "dialogues": dialogues,
        "assistant_turns": assistant_turns,
        "mean_assistant_answer_words": round(statistics.mean(answer_lengths), 2) if answer_lengths else 0,
        "median_assistant_answer_words": statistics.median(answer_lengths) if answer_lengths else 0,
        "assistant_self_introductions": sum(self_names.values()),
        "assistant_distinct_self_names": len(self_names),
        "top_self_names": self_names.most_common(12),
        "dialogues_where_assistant_names_itself": dialogues_with_self_name,
        "user_addresses_assistant_by_name": user_addresses_assistant_by_name,
        "first_token_verdict_share": round(verdict_share, 4),
        "top_first_tokens": first_tokens.most_common(10),
    }


def audit_instruction_phrases(paths: list[Path]) -> dict:
    """How often the training data contains an explicit OUTPUT-FORMAT instruction."""
    patterns = {
        "exactly one word": r"\bexactly one word\b|\bin one word\b|\bone word only\b|\bsingle word\b",
        "just a number": r"\bjust a number\b|\bonly the number\b|\bnumber only\b|\banswer with a number\b",
        "one sentence": r"\bone sentence\b|\ba single sentence\b|\bin a sentence\b",
        "list three": r"\blist (?:three|3)\b|\bname (?:three|3)\b|\bgive (?:me )?(?:three|3)\b",
        "n words limit": r"\b(?:in|under|no more than|at most|fewer than|less than)\s+\d{1,4}\s+words\b",
        "bullet/numbered": r"\bbullet(?:ed)? (?:point|list)|\bnumbered list\b",
    }
    out = {}
    for path in paths:
        if not path.exists():
            out[str(path)] = {"present": False}
            continue
        counts = Counter()
        user_turns = 0
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith("User:"):
                    continue
                user_turns += 1
                for key, pattern in patterns.items():
                    if re.search(pattern, line, flags=re.I):
                        counts[key] += 1
        total = sum(counts.values())
        out[str(path)] = {
            "present": True,
            "user_turns": user_turns,
            "format_instruction_turns": total,
            "share_of_user_turns": round(total / user_turns, 5) if user_turns else 0.0,
            "by_pattern": dict(counts),
        }
    return out


def load_dialogues(path: Path, limit: int) -> list[list[tuple[str, str]]]:
    """Parse a `User:/Assistant:` corpus into lists of (user, assistant) pairs."""
    if not path.exists():
        return []
    dialogues: list[list[tuple[str, str]]] = []
    turns: list[tuple[str, str]] = []
    role: str | None = None
    buffer: list[str] = []

    def flush():
        nonlocal role, buffer
        if role and buffer:
            turns.append((role, " ".join(buffer).strip()))
        role, buffer = None, []

    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.strip() == "<|endoftext|>":
                flush()
                pairs = []
                pending = None
                for who, text in turns:
                    if who == "user":
                        pending = text
                    elif pending is not None:
                        pairs.append((pending, text))
                        pending = None
                if len(pairs) >= 5:
                    dialogues.append(pairs)
                turns = []
                if len(dialogues) >= limit:
                    return dialogues
                continue
            if line.startswith("User:"):
                flush()
                role, buffer = "user", [line[len("User:"):].strip()]
            elif line.startswith("Assistant:"):
                flush()
                role, buffer = "assistant", [line[len("Assistant:"):].strip()]
            elif line.strip():
                buffer.append(line.strip())
    return dialogues


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------

class Runner:
    def __init__(self, assistant: Assistant, verbose: bool = True):
        self.assistant = assistant
        self.verbose = verbose
        self.records: list[dict] = []
        self.route_counts: Counter = Counter()
        self.started = time.time()

    def ask(self, suite: str, arm: str, repeat: int, turn: int, question: str,
            history: list, mode: str = "chat", probes: tuple = (), history_cap: int | None = None):
        route = self.assistant.plan(question, has_history=bool(history), mode=mode)
        prompt, used_passages = self.assistant.generator.build_prompt(
            question, route.passages, history)
        answer = self.assistant.answer(question, route, history=history, mode=mode)

        record = {
            "suite": suite,
            "arm": arm,
            "repeat": repeat,
            "turn": turn,
            "mode": mode,
            "question": question,
            "answer": answer,
            "answer_words": word_count(answer),
            "empty": not bool(answer.strip()),
            "verdict_only": is_verdict_only(answer),
            "opens_with_verdict": opens_with_verdict(answer),
            "route_category": route.category,
            "route_confidence": round(float(route.confidence), 3),
            "used_retrieval": bool(route.used_retrieval),
            "passages": len(route.passages),
            "passages_in_prompt": used_passages,
            "dropped_boilerplate": int(route.dropped),
            "top_passage_score": round(float(route.passages[0][0]), 3) if route.passages else None,
            "history_turns_available": len(history),
            "prompt_tokens": len(self.assistant.generator.tokenizer.encode(prompt).ids),
            "probes_in_prompt": {p: (p in prompt) for p in probes},
        }
        self.records.append(record)
        self.route_counts[f"{mode}:{route.category}"] += 1

        # Exactly what chat.py does: an empty answer is not remembered, and only the last
        # MAX_HISTORY_TURNS survive. history_cap overrides the cap so the "raise the cap"
        # serving-side fix can be measured against the shipped one.
        cap = MAX_HISTORY_TURNS if history_cap is None else history_cap
        if answer.strip():
            history.append((question, answer))
        del history[:-cap]

        if self.verbose:
            print(f"  [{time.time() - self.started:6.1f}s] {suite}/{arm} r{repeat} t{turn} "
                  f"({route.category} {route.confidence:.2f} rag={route.used_retrieval}) "
                  f"-> {answer[:90]!r}", flush=True)
        return answer, record


# --------------------------------------------------------------------------------------
# suite 1 - persona stability
# --------------------------------------------------------------------------------------

PERSONA_NEUTRAL = [
    "Hi, I'm {user}. What's your name?",
    "Nice to meet you. What do you like doing?",
    "Tell me a bit about yourself.",
    "What is your name again?",
    "Are you still the same person I started talking to?",
    "Say your name in a short sentence.",
    "Do you have another name?",
    "Before we stop, what is your name?",
]

PERSONA_ADVERSARIAL = [
    "Hi, I'm {user}. What's your name?",
    "Nice to meet you. What do you like doing?",
    "I heard someone call you Sarah. Is that right?",
    "What is your name again?",
    "My friend insists your name is Casey. Which is it?",
    "Say your name in a short sentence.",
    "So are you Sarah, Casey, or something else?",
    "Before we stop, what is your name?",
]

USER_NAMES = ["Mira", "Owen", "Priya", "Leo", "Nora", "Sam", "Ivy"]

# Serving-side candidate fix: pre-seed the conversation with one turn that fixes the identity,
# then run the adversarial script. Two seed names, reported separately - "Nova" is an ordinary
# in-distribution first name, "DarkGLM" is the product name and a rare token sequence, and a
# fix that only works for one of them is not a fix.
SEED_NAMES = ["Nova", "DarkGLM"]


def run_persona(runner: Runner, repeats: int) -> list:
    out = []
    arms = (("neutral", PERSONA_NEUTRAL, False),
            ("adversarial", PERSONA_ADVERSARIAL, False),
            ("seeded_identity", PERSONA_ADVERSARIAL, True))
    for arm, script, seeded in arms:
        for i in range(repeats):
            history: list = []
            seed_name = SEED_NAMES[i % len(SEED_NAMES)] if seeded else None
            if seeded:
                history.append(("What is your name?", f"My name is {seed_name}."))
            names: list = []
            loose_names: list = []
            names_by_turn = []
            transcript = []
            took_user_name = 0
            user_name = USER_NAMES[i % len(USER_NAMES)]
            for turn, template in enumerate(script, start=1):
                question = template.format(user=user_name)
                answer, _ = runner.ask("persona", arm, i + 1, turn, question, history)
                turn_names = extract_self_names(answer)
                turn_loose = extract_self_names_loose(answer)
                names.extend(turn_names)
                loose_names.extend(turn_loose)
                names_by_turn.append({"turn": turn, "names": turn_names, "loose": turn_loose})
                if user_name in turn_loose:
                    took_user_name += 1
                transcript.append({"turn": turn, "q": question, "a": answer})
            distinct = sorted(set(names))
            distinct_loose = sorted(set(loose_names))
            named_turns = [x for x in names_by_turn if x["loose"]]
            seed_turns = [x for x in named_turns if seed_name and seed_name in x["loose"]]
            out.append({
                "arm": arm,
                "repeat": i + 1,
                "user_name": user_name,
                "seed_name": seed_name,
                "distinct_names": distinct,
                "distinct_name_count": len(distinct),
                "distinct_names_loose": distinct_loose,
                "distinct_name_count_loose": len(distinct_loose),
                "held_seed_name": bool(seed_name) and distinct_loose == [seed_name],
                "seed_name_ever_repeated": bool(seed_name) and seed_name in distinct_loose,
                "seed_name_share_of_named_turns": (
                    round(len(seed_turns) / len(named_turns), 3) if named_turns else None),
                "name_mentions": len(names),
                "turns_naming_self": len(named_turns),
                "took_user_name_turns": took_user_name,
                "names_by_turn": names_by_turn,
                "transcript": transcript,
            })
    return out


# --------------------------------------------------------------------------------------
# suite 2 - coreference
# --------------------------------------------------------------------------------------

COREF_CASES = [
    ("cat", "Luna", "her", "she"),
    ("dog", "Rex", "his", "he"),
    ("rabbit", "Mochi", "her", "she"),
    ("parrot", "Kiwi", "her", "she"),
    ("hamster", "Biscuit", "his", "he"),
    ("turtle", "Pebble", "her", "she"),
]


def run_coref(runner: Runner, repeats: int) -> list:
    """Two registers, reported separately.

    terse  - the probe as specified in the brief: flat statements, then "what is her name?".
    chatty - the same fact in SODA/PersonaChat register (the shape the model trained on),
             so a failure cannot be blamed on a probe written in the wrong voice.
    """
    out = []
    for i in range(repeats):
        animal, name, possessive, subject = COREF_CASES[i % len(COREF_CASES)]
        object_pronoun = "her" if subject == "she" else "him"

        terse = [
            f"I have a {animal} called {name}.",
            f"{subject.capitalize()} likes sleeping near the window.",
            "Let's talk about hobbies for a moment.",
            "I also bought new food yesterday.",
            f"What is {possessive} name?",
        ]
        chatty = [
            f"Hey! I just adopted a {animal} and I named {object_pronoun} {name}. "
            f"{subject.capitalize()}'s so cute.",
            f"Yeah, {subject} sleeps by the window all day. It's adorable.",
            "Anyway, how has your week been going?",
            "That sounds nice. Mine has been pretty busy with work.",
            f"By the way, do you remember what I called my {animal}?",
        ]

        for arm, script in (("terse", terse), ("chatty", chatty)):
            history: list = []
            transcript = []
            final_answer = ""
            final_record = None
            for turn, question in enumerate(script, start=1):
                answer, record = runner.ask("coref", arm, i + 1, turn, question, history,
                                            probes=(name,))
                transcript.append({"turn": turn, "q": question, "a": answer})
                if turn == len(script):
                    final_answer, final_record = answer, record
            out.append({
                "arm": arm,
                "repeat": i + 1,
                "animal": animal,
                "expected": name,
                "answer": final_answer,
                "ok": name.lower() in norm(final_answer),
                "fact_in_prompt": bool(final_record["probes_in_prompt"][name]),
                "history_turns_available": final_record["history_turns_available"],
                "transcript": transcript,
            })
    return out


# --------------------------------------------------------------------------------------
# suite 3 - contradiction tracking
# --------------------------------------------------------------------------------------

CONTRADICTION_CASES = [
    ("favourite colour", "blue", "red"),
    ("cat's name", "Luna", "Milo"),
    ("home city", "Paris", "Rome"),
    ("lucky number", "seven", "four"),
    ("favourite fruit", "apple", "mango"),
    ("car colour", "green", "white"),
]


def run_contradiction(runner: Runner, repeats: int) -> list:
    out = []
    for i in range(repeats):
        key, old, new = CONTRADICTION_CASES[i % len(CONTRADICTION_CASES)]
        history: list = []
        transcript = []
        answers = []
        records = []
        script = [
            f"My {key} is {old}.",
            f"What is my {key}?",
            f"Actually I changed my mind, my {key} is {new} now.",
            f"What is my {key} now?",
        ]
        for turn, question in enumerate(script, start=1):
            answer, record = runner.ask("contradiction", "single", i + 1, turn, question,
                                        history, probes=(old, new))
            transcript.append({"turn": turn, "q": question, "a": answer})
            answers.append(answer)
            records.append(record)

        before = norm(answers[1])
        after = norm(answers[3])
        out.append({
            "repeat": i + 1,
            "key": key,
            "old": old,
            "new": new,
            "before_answer": answers[1],
            "after_answer": answers[3],
            "before_ok": old.lower() in before,
            "after_ok": new.lower() in after and old.lower() not in after,
            "after_stale": old.lower() in after,
            "after_mentions_neither": new.lower() not in after and old.lower() not in after,
            "both_facts_in_prompt": (records[3]["probes_in_prompt"][old]
                                     and records[3]["probes_in_prompt"][new]),
            "transcript": transcript,
        })
    return out


# --------------------------------------------------------------------------------------
# suite 4 - format constraints
# --------------------------------------------------------------------------------------

def _check_one_word(answer: str) -> dict:
    w = words(answer)
    return {"format_ok": len(w) == 1,
            "content_ok": bool(w) and w[0].lower() in FRUITS}


def _check_three_colours(answer: str) -> dict:
    strict, loose = three_colour_check(answer)
    return {"format_ok": strict, "content_ok": loose}


def _check_number(answer: str) -> dict:
    bare, value = bare_number(answer)
    return {"format_ok": bare, "content_ok": value == 2}


def _check_one_sentence(answer: str) -> dict:
    count = sentence_count(answer)
    return {"format_ok": count == 1,
            "content_ok": "lighthouse" in norm(answer) or "light" in norm(answer)}


FORMAT_SPECS = [
    ("one_word", "Answer in exactly one word: name a fruit.", _check_one_word),
    ("three_colours", "List three colours. Only the colours, nothing else.", _check_three_colours),
    ("number_only", "Answer with just a number: how many wheels does a bicycle have?", _check_number),
    ("one_sentence", "Write one sentence about a lighthouse.", _check_one_sentence),
]

# Serving-side candidate fix, no retraining: put two worked examples of the required shape in
# the conversation history. The app owns history, so this is deployable today. Uses different
# subject matter from the probe so it cannot leak the answer.
FEWSHOT_HISTORY = {
    "one_word": [("Answer in exactly one word: name a colour.", "Blue."),
                 ("Answer in exactly one word: name an animal.", "Otter.")],
    "three_colours": [("List three animals. Only the animals, nothing else.",
                       "Otter, heron, badger."),
                      ("List three fruits. Only the fruits, nothing else.",
                       "Pear, mango, plum.")],
    "number_only": [("Answer with just a number: how many days are in a week?", "7"),
                    ("Answer with just a number: how many legs does a spider have?", "8")],
    "one_sentence": [("Write one sentence about a river.",
                      "The river winds slowly through the valley."),
                     ("Write one sentence about a mountain.",
                      "The mountain is capped with snow all winter.")],
}


def run_format(runner: Runner, repeats: int) -> list:
    """Each constraint measured in three arms, never pooled.

    chat        never retrieves - the model alone.
    auto        what production does by default; retrieval may fire and eat the context.
    chat+2shot  serving-side candidate fix: two worked examples injected as history.

    format_ok and content_ok are scored separately on purpose. A one-word answer that names
    no fruit satisfies the format and answers nothing, and pooling the two would report that
    as a success.
    """
    out = []
    arms = (("chat", "chat", False), ("auto", "auto", False), ("chat+2shot", "chat", True))
    for arm, mode, fewshot in arms:
        for name, question, check in FORMAT_SPECS:
            for i in range(repeats):
                history: list = list(FEWSHOT_HISTORY[name]) if fewshot else []
                answer, record = runner.ask("format", f"{name}:{arm}", i + 1, 1, question,
                                            history, mode=mode)
                result = check(answer)
                out.append({
                    "constraint": name,
                    "arm": arm,
                    "mode": mode,
                    "repeat": i + 1,
                    "question": question,
                    "answer": answer,
                    "answer_words": record["answer_words"],
                    "format_ok": result["format_ok"],
                    "content_ok": result["content_ok"],
                    "both_ok": bool(result["format_ok"] and result["content_ok"]),
                    "route_category": record["route_category"],
                    "route_confidence": record["route_confidence"],
                    "used_retrieval": record["used_retrieval"],
                    "passages": record["passages"],
                    "top_passage_score": record["top_passage_score"],
                    "empty": record["empty"],
                    "verdict_only": record["verdict_only"],
                })
    return out


# --------------------------------------------------------------------------------------
# suite 5 - degradation with conversation length
# --------------------------------------------------------------------------------------

DEG_TOPICS = ["gardening", "music", "cooking", "travel", "painting", "cycling"]

# Asked identically at position 1 (fresh conversation) and at position 15+ (after 14 turns).
# Same wording in both conditions, so the comparison is position, not question.
DEG_PROBES = [
    ("one_word", "Answer in exactly one word: name a fruit."),
    ("persona", "What is your name?"),
    ("open", "Tell me one simple tip about {topic}."),
    ("longrange_coref", "What did I say my cat is called?"),
]

DEG_FILLERS = [
    "I have a cat called Luna, by the way.",
    "Let's have a casual chat about {topic}.",
    "What do you like about it?",
    "What is a common beginner mistake?",
    "How can someone practice?",
    "What tool helps most?",
    "Give a tiny example.",
    "What should I avoid?",
    "How would you explain it to a child?",
    "What is one benefit?",
    "What is one drawback?",
    "Name a related hobby.",
    "Give one encouraging sentence.",
    "What is your favourite part of it?",
]


def run_degradation(runner: Runner, repeats: int) -> dict:
    early, late, natural = [], [], []
    for i in range(repeats):
        topic = DEG_TOPICS[i % len(DEG_TOPICS)]

        # EARLY: each probe is turn 1 of its own fresh conversation. The long-range coref probe
        # needs its fact, so it gets a two-turn conversation - and is compared only against its
        # own late counterpart, never against the other probes.
        for key, template in DEG_PROBES:
            question = template.format(topic=topic)
            history: list = []
            if key == "longrange_coref":
                runner.ask("degradation", "early_setup", i + 1, 0,
                           "I have a cat called Luna, by the way.", history)
            answer, record = runner.ask("degradation", f"early:{key}", i + 1, 1, question,
                                        history, probes=("Luna",))
            early.append({"probe": key, "repeat": i + 1, "position": record["history_turns_available"] + 1,
                          "question": question, "answer": answer,
                          "words": record["answer_words"], "empty": record["empty"],
                          "verdict_only": record["verdict_only"],
                          "fact_in_prompt": record["probes_in_prompt"]["Luna"],
                          "prompt_tokens": record["prompt_tokens"]})

        # LATE: 14 filler turns, then the same probes at positions 15-18.
        history = []
        previous = ""
        for turn, template in enumerate(DEG_FILLERS, start=1):
            question = template.format(topic=topic)
            answer, record = runner.ask("degradation", "filler", i + 1, turn, question, history)
            ratio = similar(previous, answer) if previous else 0.0
            natural.append({"repeat": i + 1, "turn": turn, "question": question,
                            "answer": answer, "words": record["answer_words"],
                            "empty": record["empty"], "verdict_only": record["verdict_only"],
                            "repeat_prev_ratio": round(ratio, 3), "repeat_prev": ratio > 0.7,
                            "prompt_tokens": record["prompt_tokens"],
                            "history_turns_available": record["history_turns_available"]})
            previous = answer

        for offset, (key, template) in enumerate(DEG_PROBES):
            question = template.format(topic=topic)
            answer, record = runner.ask("degradation", f"late:{key}", i + 1,
                                        len(DEG_FILLERS) + 1 + offset, question, history,
                                        probes=("Luna",))
            late.append({"probe": key, "repeat": i + 1,
                         "position": record["history_turns_available"] + 1,
                         "question": question, "answer": answer,
                         "words": record["answer_words"], "empty": record["empty"],
                         "verdict_only": record["verdict_only"],
                         "fact_in_prompt": record["probes_in_prompt"]["Luna"],
                         "prompt_tokens": record["prompt_tokens"]})

    return {"early": early, "late": late, "natural": natural}


# Serving-side candidate fix for long-range memory: chat.py caps history at 8 turns while the
# measured prompt at turn 15 is only ~430 tokens of a 672-token budget, so the cap - not the
# budget - is what deletes turn 1. This arm repeats the same 14-turn conversation with the cap
# raised and asks the same question.
RAISED_CAP = 16


def run_raised_cap(runner: Runner, repeats: int) -> list:
    out = []
    probe = "What did I say my cat is called?"
    for i in range(repeats):
        topic = DEG_TOPICS[i % len(DEG_TOPICS)]
        history: list = []
        for turn, template in enumerate(DEG_FILLERS, start=1):
            runner.ask("raised_cap", "filler", i + 1, turn, template.format(topic=topic),
                       history, history_cap=RAISED_CAP)
        answer, record = runner.ask("raised_cap", f"late:{probe}", i + 1, len(DEG_FILLERS) + 1,
                                    probe, history, probes=("Luna",), history_cap=RAISED_CAP)
        out.append({
            "repeat": i + 1,
            "cap": RAISED_CAP,
            "position": record["history_turns_available"] + 1,
            "answer": answer,
            "correct": "luna" in norm(answer),
            "fact_in_prompt": record["probes_in_prompt"]["Luna"],
            "prompt_tokens": record["prompt_tokens"],
            "history_turns_available": record["history_turns_available"],
        })
    return out


# --------------------------------------------------------------------------------------
# suite 6 - in-distribution replay control
# --------------------------------------------------------------------------------------

def run_replay(runner: Runner, per_corpus: int) -> list:
    """Real held-out dialogues, real prior turns as history, real next user turn as the question.

    This is the register control. If the hand-written suites look broken and this one does not,
    the probes are out of distribution rather than the model being broken (onboarding section 9).
    """
    out = []
    for corpus, path in (("soda", SODA_VAL), ("personachat", PERSONACHAT_VAL)):
        dialogues = load_dialogues(path, per_corpus)
        for index, pairs in enumerate(dialogues, start=1):
            for depth_name, depth in (("early_1_prior", 1), ("late_4_prior", 4)):
                if len(pairs) <= depth:
                    continue
                history = [tuple(p) for p in pairs[:depth]]
                question, reference = pairs[depth]
                answer, record = runner.ask("replay", f"{corpus}:{depth_name}", index, depth + 1,
                                            question, list(history))
                out.append({
                    "corpus": corpus,
                    "depth": depth_name,
                    "dialogue": index,
                    "question": question,
                    "answer": answer,
                    "reference": reference,
                    "words": record["answer_words"],
                    "reference_words": word_count(reference),
                    "empty": record["empty"],
                    "verdict_only": record["verdict_only"],
                    "opens_with_verdict": record["opens_with_verdict"],
                    "similarity_to_reference": round(similar(reference, answer), 3),
                    "prompt_tokens": record["prompt_tokens"],
                })
    return out


# --------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------

def rate(rows: list, key: str) -> float:
    return round(sum(1 for r in rows if r.get(key)) / len(rows), 3) if rows else 0.0


def mean(rows: list, key: str) -> float:
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.mean(values), 2) if values else 0.0


def first_where(rows: list, key: str, want: bool = False):
    return next((r for r in rows if bool(r.get(key)) is want), None)


def summarize(results: dict, runner: Runner, audits: dict, elapsed: float, repeats: int) -> dict:
    persona = results["persona"]
    coref = results["coref"]
    contradiction = results["contradiction"]
    fmt = results["format"]
    deg = results["degradation"]
    replay = results["replay"]

    persona_by_arm = {}
    for arm in ("neutral", "adversarial", "seeded_identity"):
        rows = [r for r in persona if r["arm"] == arm]
        if not rows:
            continue
        entry = {
            "n_conversations": len(rows),
            "turns_per_conversation": len(PERSONA_NEUTRAL),
            "mean_distinct_self_names_strict": round(statistics.mean(
                r["distinct_name_count"] for r in rows), 2),
            "mean_distinct_self_names_incl_its_x": round(statistics.mean(
                r["distinct_name_count_loose"] for r in rows), 2),
            "max_distinct_self_names": max(r["distinct_name_count_loose"] for r in rows),
            "unstable_rate_2plus_names": rate(
                [{"x": r["distinct_name_count_loose"] > 1} for r in rows], "x"),
            "never_names_itself_rate": rate(
                [{"x": r["distinct_name_count_loose"] == 0} for r in rows], "x"),
            "conversations_taking_the_user_name": sum(
                1 for r in rows if r["took_user_name_turns"] > 0),
            "mean_turns_naming_self_of_8": round(statistics.mean(
                r["turns_naming_self"] for r in rows), 2),
            "names_seen": sorted({n for r in rows for n in r["distinct_names_loose"]}),
            "example_unstable": next(
                (r for r in rows if r["distinct_name_count_loose"] > 1), None),
        }
        if arm == "seeded_identity":
            entry["held_seed_name_only_rate"] = rate(rows, "held_seed_name")
            entry["seed_name_ever_repeated_rate"] = rate(rows, "seed_name_ever_repeated")
            shares = [r["seed_name_share_of_named_turns"] for r in rows
                      if r["seed_name_share_of_named_turns"] is not None]
            entry["mean_seed_name_share_of_named_turns"] = (
                round(statistics.mean(shares), 3) if shares else None)
            entry["by_seed_name"] = {
                seed: {
                    "n": len([r for r in rows if r["seed_name"] == seed]),
                    "held_seed_name_only_rate": rate(
                        [r for r in rows if r["seed_name"] == seed], "held_seed_name"),
                    "seed_name_ever_repeated_rate": rate(
                        [r for r in rows if r["seed_name"] == seed], "seed_name_ever_repeated"),
                    "mean_distinct_names": round(statistics.mean(
                        [r["distinct_name_count_loose"] for r in rows
                         if r["seed_name"] == seed]), 2)
                    if any(r["seed_name"] == seed for r in rows) else None,
                }
                for seed in SEED_NAMES
            }
        persona_by_arm[arm] = entry

    coref_by_arm = {}
    for arm in ("terse", "chatty"):
        rows = [r for r in coref if r["arm"] == arm]
        coref_by_arm[arm] = {
            "n": len(rows),
            "correct_recall_rate": rate(rows, "ok"),
            "fact_in_prompt_rate": rate(rows, "fact_in_prompt"),
            "failure_example": first_where(rows, "ok", False),
        }

    format_by = {}
    for name, _, _ in FORMAT_SPECS:
        for arm in ("chat", "auto", "chat+2shot"):
            rows = [r for r in fmt if r["constraint"] == name and r["arm"] == arm]
            if not rows:
                continue
            format_by[f"{name}:{arm}"] = {
                "n": len(rows),
                "format_ok_rate": rate(rows, "format_ok"),
                "content_ok_rate": rate(rows, "content_ok"),
                "format_and_content_ok_rate": rate(rows, "both_ok"),
                "verdict_only_rate": rate(rows, "verdict_only"),
                "mean_answer_words": mean(rows, "answer_words"),
                "retrieval_rate": rate(rows, "used_retrieval"),
                "route_categories": dict(Counter(r["route_category"] for r in rows)),
                "failure_example": first_where(rows, "format_ok", False),
            }

    deg_by_probe = {}
    for key, _ in DEG_PROBES:
        early = [r for r in deg["early"] if r["probe"] == key]
        late = [r for r in deg["late"] if r["probe"] == key]
        deg_by_probe[key] = {
            "n_each": len(early),
            "early_mean_words": mean(early, "words"),
            "late_mean_words": mean(late, "words"),
            "early_empty_rate": rate(early, "empty"),
            "late_empty_rate": rate(late, "empty"),
            "early_verdict_only_rate": rate(early, "verdict_only"),
            "late_verdict_only_rate": rate(late, "verdict_only"),
            "early_fact_in_prompt_rate": rate(early, "fact_in_prompt"),
            "late_fact_in_prompt_rate": rate(late, "fact_in_prompt"),
            "early_mean_prompt_tokens": mean(early, "prompt_tokens"),
            "late_mean_prompt_tokens": mean(late, "prompt_tokens"),
        }
    deg_by_probe["one_word"]["early_format_ok_rate"] = rate(
        [{"x": word_count(r["answer"]) == 1} for r in deg["early"] if r["probe"] == "one_word"], "x")
    deg_by_probe["one_word"]["late_format_ok_rate"] = rate(
        [{"x": word_count(r["answer"]) == 1} for r in deg["late"] if r["probe"] == "one_word"], "x")
    deg_by_probe["longrange_coref"]["early_correct_rate"] = rate(
        [{"x": "luna" in norm(r["answer"])} for r in deg["early"] if r["probe"] == "longrange_coref"], "x")
    deg_by_probe["longrange_coref"]["late_correct_rate"] = rate(
        [{"x": "luna" in norm(r["answer"])} for r in deg["late"] if r["probe"] == "longrange_coref"], "x")

    natural_by_turn = {}
    by_turn = defaultdict(list)
    for row in deg["natural"]:
        by_turn[row["turn"]].append(row)
    for turn in sorted(by_turn):
        rows = by_turn[turn]
        natural_by_turn[f"turn_{turn:02d}"] = {
            "n": len(rows),
            "mean_words": mean(rows, "words"),
            "empty_rate": rate(rows, "empty"),
            "repeat_prev_rate": rate(rows, "repeat_prev"),
            "mean_prompt_tokens": mean(rows, "prompt_tokens"),
            "mean_history_turns": mean(rows, "history_turns_available"),
        }

    replay_by = {}
    for corpus in ("soda", "personachat"):
        for depth in ("early_1_prior", "late_4_prior"):
            rows = [r for r in replay if r["corpus"] == corpus and r["depth"] == depth]
            if not rows:
                continue
            replay_by[f"{corpus}:{depth}"] = {
                "n": len(rows),
                "mean_words": mean(rows, "words"),
                "reference_mean_words": mean(rows, "reference_words"),
                "empty_rate": rate(rows, "empty"),
                "verdict_only_rate": rate(rows, "verdict_only"),
                "opens_with_verdict_rate": rate(rows, "opens_with_verdict"),
                "mean_similarity_to_reference": mean(rows, "similarity_to_reference"),
            }

    all_records = runner.records
    return {
        "run": {
            "repeats": repeats,
            "elapsed_seconds": round(elapsed, 1),
            "total_generations": len(all_records),
            "checkpoint": runner.assistant.generator.checkpoint,
            "tokenizer": str(runner.assistant.generator.tokenizer_path),
            "max_history_turns": MAX_HISTORY_TURNS,
            "temperature_by_category": config.TEMPERATURE_BY_CATEGORY,
            "echo_penalty": config.ECHO_PENALTY,
            "repetition_penalty": config.REPETITION_PENALTY,
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "context_length": runner.assistant.generator.config.context_length,
        },
        "global": {
            "empty_answer_rate": rate(all_records, "empty"),
            "verdict_only_rate": rate(all_records, "verdict_only"),
            "opens_with_verdict_rate": rate(all_records, "opens_with_verdict"),
            "route_counts": dict(runner.route_counts),
        },
        "persona_stability": persona_by_arm,
        "coreference": coref_by_arm,
        "contradiction_tracking": {
            "n": len(contradiction),
            "initial_fact_recall_rate": rate(contradiction, "before_ok"),
            "updated_fact_recall_rate": rate(contradiction, "after_ok"),
            "stale_after_update_rate": rate(contradiction, "after_stale"),
            "mentions_neither_rate": rate(contradiction, "after_mentions_neither"),
            "both_facts_in_prompt_rate": rate(contradiction, "both_facts_in_prompt"),
            "failure_example": first_where(contradiction, "after_ok", False),
        },
        "format_constraints": format_by,
        "degradation_position_controlled": deg_by_probe,
        "degradation_natural_by_turn": natural_by_turn,
        "history_cap_ab": {
            "shipped_cap_8": {
                "n": len([r for r in deg["late"] if r["probe"] == "longrange_coref"]),
                "correct_rate": deg_by_probe["longrange_coref"]["late_correct_rate"],
                "fact_in_prompt_rate": deg_by_probe["longrange_coref"]["late_fact_in_prompt_rate"],
                "mean_prompt_tokens": deg_by_probe["longrange_coref"]["late_mean_prompt_tokens"],
            },
            f"raised_cap_{RAISED_CAP}": {
                "n": len(results["raised_cap"]),
                "correct_rate": rate(results["raised_cap"], "correct"),
                "fact_in_prompt_rate": rate(results["raised_cap"], "fact_in_prompt"),
                "mean_prompt_tokens": mean(results["raised_cap"], "prompt_tokens"),
                "mean_history_turns": mean(results["raised_cap"], "history_turns_available"),
            },
            "context_budget_tokens": (runner.assistant.generator.config.context_length
                                      - config.ANSWER_TOKEN_RESERVE),
        },
        "in_distribution_replay": replay_by,
        "corpus_audit": audits,
    }


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------

def write_report(path: Path, summary: dict, results: dict, runner: Runner) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", errors="replace") as f:
        f.write("=" * 88 + "\n")
        f.write("STRESS TEST - MULTI-TURN CONVERSATION AND INSTRUCTION FOLLOWING\n")
        f.write("tools/stress_conversation.py   (question answering is covered by stress_qa.py)\n")
        f.write("=" * 88 + "\n\n")
        f.write("SEGMENTATION - these populations are reported separately and never pooled:\n")
        f.write("  persona:      neutral / adversarial (a name is suggested) / seeded_identity\n")
        f.write("                (serving-side fix: identity pre-seeded in history), and the\n")
        f.write("                seeded arm is split again by seed name (Nova vs DarkGLM)\n")
        f.write("  coreference:  terse probe register vs SODA/PersonaChat chatty register\n")
        f.write("  format:       per constraint type, and chat / auto / chat+2shot arms;\n")
        f.write("                format_ok and content_ok are scored separately\n")
        f.write("  degradation:  identical probe at position 1 vs position 15+, plus natural\n")
        f.write("                turns, plus a history-cap A/B (8 vs 16 turns)\n")
        f.write("  replay:       SODA vs PersonaChat, 1 prior turn vs 4 prior turns\n\n")
        f.write("SUMMARY JSON\n")
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n\n")

        f.write("=" * 88 + "\nRAW TRANSCRIPTS\n" + "=" * 88 + "\n")

        f.write("\n--- SUITE 1: PERSONA STABILITY ---\n")
        for row in results["persona"]:
            f.write(f"\n[{row['arm']}] repeat {row['repeat']} user={row['user_name']} "
                    f"seed={row['seed_name']} "
                    f"distinct_self_names={row['distinct_name_count_loose']} "
                    f"{row['distinct_names_loose']}\n")
            for turn in row["transcript"]:
                f.write(f"  {turn['turn']:>2} U: {turn['q']}\n")
                f.write(f"     A: {turn['a']}\n")

        f.write("\n--- SUITE 2: COREFERENCE ---\n")
        for row in results["coref"]:
            f.write(f"\n[{row['arm']}] repeat {row['repeat']} expected={row['expected']} "
                    f"ok={row['ok']} fact_in_prompt={row['fact_in_prompt']}\n")
            for turn in row["transcript"]:
                f.write(f"  {turn['turn']:>2} U: {turn['q']}\n")
                f.write(f"     A: {turn['a']}\n")

        f.write("\n--- SUITE 3: CONTRADICTION TRACKING ---\n")
        for row in results["contradiction"]:
            f.write(f"\nrepeat {row['repeat']} {row['key']}: {row['old']} -> {row['new']} "
                    f"before_ok={row['before_ok']} after_ok={row['after_ok']} "
                    f"stale={row['after_stale']} both_in_prompt={row['both_facts_in_prompt']}\n")
            for turn in row["transcript"]:
                f.write(f"  {turn['turn']:>2} U: {turn['q']}\n")
                f.write(f"     A: {turn['a']}\n")

        f.write("\n--- SUITE 4: FORMAT CONSTRAINTS ---\n")
        for row in results["format"]:
            f.write(f"\n[{row['constraint']}/{row['arm']}] repeat {row['repeat']} "
                    f"format_ok={row['format_ok']} content_ok={row['content_ok']} "
                    f"route={row['route_category']} {row['route_confidence']} "
                    f"rag={row['used_retrieval']} passages={row['passages']}\n")
            f.write(f"  U: {row['question']}\n")
            f.write(f"  A: {row['answer']}\n")

        f.write("\n--- SUITE 5a: DEGRADATION, POSITION-CONTROLLED (same question, two positions) ---\n")
        for condition in ("early", "late"):
            for row in results["degradation"][condition]:
                f.write(f"\n[{condition}:{row['probe']}] repeat {row['repeat']} "
                        f"position={row['position']} words={row['words']} "
                        f"fact_in_prompt={row['fact_in_prompt']} "
                        f"prompt_tokens={row['prompt_tokens']}\n")
                f.write(f"  U: {row['question']}\n")
                f.write(f"  A: {row['answer']}\n")

        f.write("\n--- SUITE 5b: DEGRADATION, NATURAL 14-TURN CONVERSATIONS ---\n")
        for row in results["degradation"]["natural"]:
            f.write(f"\nrepeat {row['repeat']} turn {row['turn']:>2} words={row['words']} "
                    f"repeat_prev_ratio={row['repeat_prev_ratio']} "
                    f"history={row['history_turns_available']} "
                    f"prompt_tokens={row['prompt_tokens']}\n")
            f.write(f"  U: {row['question']}\n")
            f.write(f"  A: {row['answer']}\n")

        f.write("\n--- SUITE 5c: HISTORY CAP A/B (same conversation, cap 8 vs cap "
                f"{RAISED_CAP}) ---\n")
        for row in results["raised_cap"]:
            f.write(f"\nrepeat {row['repeat']} cap={row['cap']} position={row['position']} "
                    f"correct={row['correct']} fact_in_prompt={row['fact_in_prompt']} "
                    f"history_turns={row['history_turns_available']} "
                    f"prompt_tokens={row['prompt_tokens']}\n")
            f.write("  U: What did I say my cat is called?\n")
            f.write(f"  A: {row['answer']}\n")

        f.write("\n--- SUITE 6: IN-DISTRIBUTION REPLAY (real held-out dialogues) ---\n")
        for row in results["replay"]:
            f.write(f"\n[{row['corpus']}/{row['depth']}] dialogue {row['dialogue']} "
                    f"words={row['words']} reference_words={row['reference_words']} "
                    f"similarity={row['similarity_to_reference']}\n")
            f.write(f"  U:   {row['question']}\n")
            f.write(f"  A:   {row['answer']}\n")
            f.write(f"  REF: {row['reference']}\n")

        f.write("\n\n" + "=" * 88 + "\nALL GENERATION RECORDS (JSON)\n" + "=" * 88 + "\n")
        f.write(json.dumps(runner.records, ensure_ascii=False, indent=1))
        f.write("\n")


# --------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=8, help="N per probe, must be >= 5")
    parser.add_argument("--persona-repeats", type=int, default=8,
                        help="persona is the headline metric and gets a larger N")
    parser.add_argument("--deg-repeats", type=int, default=5,
                        help="degradation conversations are 15+ turns each, so N is smaller")
    parser.add_argument("--replay-per-corpus", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--audit-only", action="store_true", help="corpus counts only, no GPU")
    args = parser.parse_args()
    if args.repeats < 5:
        raise SystemExit("--repeats must be >= 5: this project does not conclude from small N")

    start = time.time()
    print("Auditing training corpora...", flush=True)
    audits = {
        "soda_train": audit_dialogue_corpus(SODA_TRAIN),
        "personachat_train": audit_dialogue_corpus(PERSONACHAT_TRAIN),
        "no_robots_train": audit_dialogue_corpus(NO_ROBOTS_TRAIN),
        "oasst1_train": audit_dialogue_corpus(OASST1_TRAIN),
        "format_instructions": audit_instruction_phrases(
            [SODA_TRAIN, PERSONACHAT_TRAIN, NO_ROBOTS_TRAIN, OASST1_TRAIN]),
    }
    print(json.dumps(audits, ensure_ascii=False, indent=2), flush=True)

    if args.audit_only:
        print(f"\naudit only, {round(time.time() - start, 1)}s")
        return 0

    torch.manual_seed(args.seed)
    print("\nLoading assistant...", flush=True)
    assistant = Assistant(device=args.device, quiet=True)
    runner = Runner(assistant)
    try:
        results = {}
        print("\n== suite 1: persona stability ==", flush=True)
        results["persona"] = run_persona(runner, max(args.persona_repeats, args.repeats))
        print("\n== suite 2: coreference ==", flush=True)
        results["coref"] = run_coref(runner, args.repeats)
        print("\n== suite 3: contradiction tracking ==", flush=True)
        results["contradiction"] = run_contradiction(runner, args.repeats)
        print("\n== suite 4: format constraints ==", flush=True)
        results["format"] = run_format(runner, args.repeats)
        print("\n== suite 5: degradation ==", flush=True)
        results["degradation"] = run_degradation(runner, args.deg_repeats)
        print(f"\n== suite 5c: history cap A/B (cap {RAISED_CAP}) ==", flush=True)
        results["raised_cap"] = run_raised_cap(runner, args.deg_repeats)
        print("\n== suite 6: in-distribution replay ==", flush=True)
        results["replay"] = run_replay(runner, args.replay_per_corpus)
        print("\n== suite 6: in-distribution replay ==", flush=True)
        results["replay"] = run_replay(runner, args.replay_per_corpus)
    finally:
        assistant.close()

    summary = summarize(results, runner, audits, time.time() - start, args.repeats)
    write_report(args.output, summary, results, runner)
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.output}  ({len(runner.records)} generations, "
          f"{round(time.time() - start, 1)}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
