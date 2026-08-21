"""Independent adversarial checks of five behavioural claims about DarkGLM."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from orchestrator import config
import orchestrator.pipeline as pipeline_module
from orchestrator.pipeline import Assistant, Route
from tools.independent_opening_verb_counts import (
    DECLARED_MIX,
    count_file,
    has_user_lines,
)


OUT_JSON = Path("tests/independent_claims_verification.json")
OUT_TEXT = Path("tests/independent_claims_verification.txt")
VERDICT_WORDS = {"yes", "no", "true", "false", "unknown"}
WORD_RE = re.compile(r"[A-Za-z0-9']+")
YES_NO_OPEN = re.compile(
    r"^\s*(?:is|are|am|was|were|do|does|did|can|could|would|should|will|"
    r"has|have|had|may|might|must|shall)\b",
    re.I,
)


def words(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def opens_with_verdict(text: str) -> bool:
    found = words(text)
    return bool(found) and found[0].lower() in VERDICT_WORDS


def verdict_only(text: str) -> bool:
    found = words(text)
    return len(found) == 1 and found[0].lower() in VERDICT_WORDS


def contains_expected(text: str, alternatives: list[str] | tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    for alternative in alternatives:
        pattern = r"(?<!\w)" + re.escape(alternative.lower()) + r"(?!\w)"
        if re.search(pattern, lowered):
            return True
    return False


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rate(rows: list[dict], key: str) -> float:
    return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else 0.0


def mean(rows: list[dict], key: str) -> float:
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else 0.0


def summarize_binary(rows: list[dict], key: str) -> dict:
    return {
        "n": len(rows),
        "hits": sum(bool(row.get(key)) for row in rows),
        "rate": round(rate(rows, key), 4),
    }


def ask(
    assistant: Assistant,
    question: str,
    *,
    history: list[tuple[str, str]] | tuple = (),
    mode: str = "auto",
    seed: int,
    route_override: Route | None = None,
) -> dict:
    set_seed(seed)
    route = route_override or assistant.plan(
        question, has_history=bool(history), mode=mode
    )
    normalized = pipeline_module.normalise_shouting(question)
    effective_history = history
    if route.used_retrieval and config.DROP_HISTORY_WHEN_RETRIEVING:
        effective_history = ()
    prompt, passages_used = assistant.generator.build_prompt(
        normalized, route.passages, effective_history
    )
    answer = assistant.answer(question, route, history=history, mode=mode)
    return {
        "question": question,
        "normalized_question": normalized,
        "answer": answer,
        "mode": mode,
        "category": route.category,
        "confidence": round(float(route.confidence), 4),
        "used_retrieval": bool(route.used_retrieval),
        "passages": len(route.passages),
        "passages_used": passages_used,
        "passage_text": [text for _, text in route.passages],
        "history_given": len(history),
        "history_in_prompt": len(effective_history),
        "prompt_tokens": len(assistant.generator.tokenizer.encode(prompt).ids),
        "prompt": prompt,
        "opens_with_verdict": opens_with_verdict(answer),
        "verdict_only": verdict_only(answer),
    }


def run_claim_1(assistant: Assistant, repeats: int) -> dict:
    questions = [
        ("What is the capital city of New Zealand?", ["wellington"]),
        ("Which planet has the Great Red Spot?", ["jupiter"]),
        ("Who painted The Starry Night?", ["van gogh", "vincent van gogh"]),
        ("In which year did the Titanic sink?", ["1912"]),
        ("What gas makes up most of Earth's atmosphere?", ["nitrogen"]),
        ("What is the largest moon of Saturn?", ["titan"]),
    ]
    histories = {
        "conversational": [
            ("How has your afternoon been?", "It has been calm, thanks for asking."),
            ("I chose tea instead of coffee today.", "Tea can be a relaxing choice."),
            ("Tell me something cheerful.", "Small acts of kindness can brighten a day."),
        ],
        "factual_sounding": [
            ("What is the freezing point of water?", "Pure water freezes at zero Celsius."),
            ("Where is Venus in the Solar System?", "Venus is the second planet from the Sun."),
            ("How many sides does a triangle have?", "A triangle has three sides."),
        ],
    }
    original = config.DROP_HISTORY_WHEN_RETRIEVING
    rows = []
    try:
        for history_name, history in histories.items():
            for drop in (False, True):
                config.DROP_HISTORY_WHEN_RETRIEVING = drop
                for q_index, (question, expected) in enumerate(questions):
                    for repeat in range(repeats):
                        row = ask(
                            assistant,
                            question,
                            history=history,
                            mode="auto",
                            seed=10_000 + q_index * 100 + repeat,
                        )
                        row.update({
                            "history_type": history_name,
                            "drop_history": drop,
                            "repeat": repeat + 1,
                            "expected": expected,
                            "correct": contains_expected(row["answer"], expected),
                            "passage_support": any(
                                contains_expected(text, expected) for text in row["passage_text"]
                            ),
                        })
                        rows.append(row)
    finally:
        config.DROP_HISTORY_WHEN_RETRIEVING = original

    summary = {}
    for history_name in histories:
        for drop in (False, True):
            segment = [
                row for row in rows
                if row["history_type"] == history_name and row["drop_history"] == drop
            ]
            summary[f"{history_name}:drop={drop}"] = {
                **summarize_binary(segment, "correct"),
                "retrieval_rate": round(rate(segment, "used_retrieval"), 4),
                "passage_support_rate": round(rate(segment, "passage_support"), 4),
                "mean_prompt_tokens": round(mean(segment, "prompt_tokens"), 1),
            }
    return {"summary": summary, "rows": rows}


def run_claim_2(assistant: Assistant, repeats: int) -> dict:
    questions = [
        ("WHICH ANIMAL IS THE FASTEST LAND ANIMAL?", ["cheetah"]),
        ("WHO COMPOSED THE FOUR SEASONS?", ["vivaldi"]),
        ("WHAT IS THE CAPITAL OF BRAZIL?", ["brasilia", "brasília"]),
        ("WHICH PLANET IS CLOSEST TO THE SUN?", ["mercury"]),
        ("WHAT GAS DO HUMANS NEED TO BREATHE?", ["oxygen"]),
        ("HOW MANY CONTINENTS ARE THERE?", ["seven", "7"]),
    ]
    original = pipeline_module.normalise_shouting
    rows = []
    try:
        for arm in ("raw_caps", "live_caps", "lowercase"):
            pipeline_module.normalise_shouting = (
                (lambda text: text) if arm == "raw_caps" else original
            )
            for q_index, (caps_question, expected) in enumerate(questions):
                question = caps_question.lower() if arm == "lowercase" else caps_question
                for repeat in range(repeats):
                    row = ask(
                        assistant,
                        question,
                        seed=20_000 + q_index * 100 + repeat,
                    )
                    row.update({
                        "arm": arm,
                        "repeat": repeat + 1,
                        "expected": expected,
                        "correct": contains_expected(row["answer"], expected),
                        "passage_support": any(
                            contains_expected(text, expected) for text in row["passage_text"]
                        ),
                        "raw_token_count": len(
                            assistant.generator.tokenizer.encode(question).ids
                        ),
                        "normalized_token_count": len(
                            assistant.generator.tokenizer.encode(
                                pipeline_module.normalise_shouting(question)
                            ).ids
                        ),
                    })
                    rows.append(row)
    finally:
        pipeline_module.normalise_shouting = original

    mixed_acronyms = [
        "What does NASA study?",
        "Where is IBM headquartered?",
        "What does HTTP stand for?",
        "What information does DNA carry?",
        "Is the UK a member of NATO?",
        "What is a GPU used for?",
    ]
    mixed_rows = []
    for index, question in enumerate(mixed_acronyms):
        outputs = {}
        for arm, normalizer in (("identity", lambda text: text), ("live", original)):
            try:
                pipeline_module.normalise_shouting = normalizer
                outputs[arm] = ask(assistant, question, seed=21_000 + index)
            finally:
                pipeline_module.normalise_shouting = original
        mixed_rows.append({
            "question": question,
            "live_transform": original(question),
            "changed": original(question) != question,
            "identity_answer": outputs["identity"]["answer"],
            "live_answer": outputs["live"]["answer"],
            "answers_identical": outputs["identity"]["answer"] == outputs["live"]["answer"],
        })

    terse_acronyms = [
        ("WHO HQ LOCATION?", ["geneva"]),
        ("UN HQ LOCATION?", ["new york"]),
        ("US CAPITAL?", ["washington"]),
        ("THE WHO SINGER?", ["daltrey"]),
        ("APPLE CEO?", ["cook"]),
        ("IBM FOUNDER?", ["watson"]),
    ]
    terse_rows = []
    for index, (question, expected) in enumerate(terse_acronyms):
        for arm, normalizer in (("identity", lambda text: text), ("live", original)):
            try:
                pipeline_module.normalise_shouting = normalizer
                row = ask(assistant, question, seed=22_000 + index)
            finally:
                pipeline_module.normalise_shouting = original
            row.update({
                "arm": arm,
                "expected": expected,
                "correct": contains_expected(row["answer"], expected),
                "changed_by_live_normalizer": original(question) != question,
            })
            terse_rows.append(row)

    summary = {}
    for arm in ("raw_caps", "live_caps", "lowercase"):
        segment = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            **summarize_binary(segment, "correct"),
            "retrieval_rate": round(rate(segment, "used_retrieval"), 4),
            "passage_support_rate": round(rate(segment, "passage_support"), 4),
            "mean_raw_tokens": round(mean(segment, "raw_token_count"), 1),
            "mean_normalized_tokens": round(mean(segment, "normalized_token_count"), 1),
        }
    for arm in ("identity", "live"):
        segment = [row for row in terse_rows if row["arm"] == arm]
        summary[f"terse_acronyms:{arm}"] = summarize_binary(segment, "correct")
    summary["mixed_acronyms"] = {
        "n": len(mixed_rows),
        "changed": sum(row["changed"] for row in mixed_rows),
        "identical_answers": sum(row["answers_identical"] for row in mixed_rows),
    }
    paired = defaultdict(dict)
    for row in rows:
        paired[(row["question"].lower(), row["repeat"])][row["arm"]] = row
    transitions = Counter()
    same_routes = 0
    same_live_lower_answers = 0
    for arms in paired.values():
        raw, live, lower = arms["raw_caps"], arms["live_caps"], arms["lowercase"]
        transitions[f"{int(raw['correct'])}->{int(live['correct'])}"] += 1
        same_routes += (
            raw["category"] == live["category"]
            and raw["passage_text"] == live["passage_text"]
        )
        same_live_lower_answers += live["answer"] == lower["answer"]
    summary["paired_caps"] = {
        "transitions_raw_to_live": dict(transitions),
        "same_route_and_passages": same_routes,
        "pairs": len(paired),
        "live_answer_equals_lowercase": same_live_lower_answers,
    }
    return {
        "summary": summary,
        "caps_rows": rows,
        "mixed_acronym_rows": mixed_rows,
        "terse_acronym_rows": terse_rows,
    }


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = (start + stop - 1) / 2 + 1
        for index in order[start:stop]:
            result[index] = average
        start = stop
    return result


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    l_mean, r_mean = statistics.mean(left), statistics.mean(right)
    numerator = sum((a - l_mean) * (b - r_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - l_mean) ** 2 for a in left) * sum((b - r_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else 0.0


def opening_counts() -> dict:
    by_dataset: dict[str, Counter] = {}
    for path in sorted(Path("data").glob("*/train.txt")):
        if has_user_lines(path):
            by_dataset[path.parent.name] = count_file(path)
    all_counts: Counter = Counter()
    for counts in by_dataset.values():
        all_counts.update(counts)

    checkpoint = Path("checkpoints/finetune_v2_best.pt")
    checkpoint_time = checkpoint.stat().st_mtime
    proxy_counts: Counter = Counter()
    proxy_datasets = []
    excluded_late = []
    for name in DECLARED_MIX:
        path = Path("data") / name / "train.txt"
        if not path.exists():
            continue
        if path.stat().st_mtime > checkpoint_time:
            excluded_late.append(name)
            continue
        proxy_counts.update(by_dataset.get(name, Counter()))
        proxy_datasets.append(name)
    return {
        "by_dataset": by_dataset,
        "all_counts": all_counts,
        "proxy_counts": proxy_counts,
        "all_user_turns": sum(all_counts.values()),
        "proxy_user_turns": sum(proxy_counts.values()),
        "proxy_datasets": proxy_datasets,
        "excluded_late": excluded_late,
        "checkpoint_mtime": checkpoint.stat().st_mtime,
    }


def run_claim_3(assistant: Assistant) -> dict:
    count_data = opening_counts()
    verbs = [
        ("explain", "Explain {thing}."),
        ("give", "Give me {thing}."),
        ("tell", "Tell me {thing}."),
        ("write", "Write {thing}."),
        ("describe", "Describe {thing}."),
        ("name", "Name {thing}."),
        ("provide", "Provide {thing}."),
        ("identify", "Identify {thing}."),
        ("state", "State {thing}."),
        ("show", "Show me {thing}."),
        ("specify", "Specify {thing}."),
        ("supply", "Supply {thing}."),
        ("report", "Report {thing}."),
        ("recall", "Recall {thing}."),
        ("mention", "Mention {thing}."),
    ]
    facts = [
        ("the capital of Australia", ["canberra"],
         "Australia's capital city is Canberra."),
        ("the author of The Hobbit", ["tolkien"],
         "The Hobbit was written by J. R. R. Tolkien."),
        ("the largest moon of Saturn", ["titan"],
         "Titan is the largest moon of Saturn."),
        ("the scientist who discovered penicillin", ["fleming"],
         "Alexander Fleming discovered penicillin."),
        ("the currency of Japan", ["yen"],
         "Japan uses the yen as its currency."),
    ]
    rows = []
    controlled_rows = []
    for verb_index, (verb, template) in enumerate(verbs):
        for fact_index, (thing, expected, controlled_passage) in enumerate(facts):
            row = ask(
                assistant,
                template.format(thing=thing),
                seed=30_000 + fact_index,
            )
            row.update({
                "verb": verb,
                "fact": thing,
                "expected": expected,
                "correct": contains_expected(row["answer"], expected),
                "passage_support": any(
                    contains_expected(text, expected) for text in row["passage_text"]
                ),
                "raw_all_count": count_data["all_counts"][verb],
                "checkpoint_proxy_count": count_data["proxy_counts"][verb],
            })
            rows.append(row)
            controlled_route = Route(
                category="instruction",
                confidence=1.0,
                used_retrieval=True,
                passages=[(1.0, controlled_passage)],
                dropped=0,
                mode="ask",
            )
            controlled = ask(
                assistant,
                template.format(thing=thing),
                mode="ask",
                seed=31_000 + fact_index,
                route_override=controlled_route,
            )
            controlled.update({
                "verb": verb,
                "fact": thing,
                "expected": expected,
                "correct": contains_expected(controlled["answer"], expected),
                "raw_all_count": count_data["all_counts"][verb],
                "checkpoint_proxy_count": count_data["proxy_counts"][verb],
            })
            controlled_rows.append(controlled)

    controls = []
    for fact_index, (thing, expected, _) in enumerate(facts):
        row = ask(assistant, f"What is {thing}?", seed=30_000 + fact_index)
        row.update({
            "fact": thing,
            "expected": expected,
            "correct": contains_expected(row["answer"], expected),
            "passage_support": any(
                contains_expected(text, expected) for text in row["passage_text"]
            ),
        })
        controls.append(row)

    by_verb = {}
    controlled_by_verb = {}
    for verb, _ in verbs:
        segment = [row for row in rows if row["verb"] == verb]
        by_verb[verb] = {
            **summarize_binary(segment, "correct"),
            "raw_all_count": count_data["all_counts"][verb],
            "checkpoint_proxy_count": count_data["proxy_counts"][verb],
            "retrieval_rate": round(rate(segment, "used_retrieval"), 4),
            "passage_support_rate": round(rate(segment, "passage_support"), 4),
        }
        supported = [row for row in segment if row["passage_support"]]
        by_verb[verb]["supported_n"] = len(supported)
        by_verb[verb]["supported_hits"] = sum(row["correct"] for row in supported)
        by_verb[verb]["supported_rate"] = round(rate(supported, "correct"), 4)
        controlled_segment = [
            row for row in controlled_rows if row["verb"] == verb
        ]
        controlled_by_verb[verb] = {
            **summarize_binary(controlled_segment, "correct"),
            "checkpoint_proxy_count": count_data["proxy_counts"][verb],
        }

    frequency = [
        math.log10(by_verb[verb]["checkpoint_proxy_count"] + 1)
        for verb, _ in verbs
    ]
    accuracy = [by_verb[verb]["rate"] for verb, _ in verbs]
    spearman = pearson(rank(frequency), rank(accuracy))
    supported_accuracy = [by_verb[verb]["supported_rate"] for verb, _ in verbs]
    supported_spearman = pearson(rank(frequency), rank(supported_accuracy))
    controlled_accuracy = [controlled_by_verb[verb]["rate"] for verb, _ in verbs]
    controlled_spearman = pearson(rank(frequency), rank(controlled_accuracy))

    bins = {"high_1000_plus": [], "mid_100_to_999": [], "low_under_100": []}
    for row in rows:
        count = row["checkpoint_proxy_count"]
        key = "high_1000_plus" if count >= 1000 else (
            "mid_100_to_999" if count >= 100 else "low_under_100"
        )
        bins[key].append(row)
    bin_summary = {
        key: {
            **summarize_binary(segment, "correct"),
            "verbs": sorted({row["verb"] for row in segment}),
            "passage_support_rate": round(rate(segment, "passage_support"), 4),
        }
        for key, segment in bins.items()
    }
    controlled_bins = {"high_1000_plus": [], "mid_100_to_999": [], "low_under_100": []}
    for row in controlled_rows:
        count = row["checkpoint_proxy_count"]
        key = "high_1000_plus" if count >= 1000 else (
            "mid_100_to_999" if count >= 100 else "low_under_100"
        )
        controlled_bins[key].append(row)
    controlled_bin_summary = {
        key: {
            **summarize_binary(segment, "correct"),
            "verbs": sorted({row["verb"] for row in segment}),
        }
        for key, segment in controlled_bins.items()
    }
    selected_counts = {
        verb: {
            "raw_all": count_data["all_counts"][verb],
            "checkpoint_proxy": count_data["proxy_counts"][verb],
        }
        for verb, _ in verbs
    }
    return {
        "summary": {
            "by_verb": by_verb,
            "controlled_context_by_verb": controlled_by_verb,
            "frequency_bins": bin_summary,
            "controlled_context_frequency_bins": controlled_bin_summary,
            "spearman_log_frequency_vs_accuracy": round(spearman, 4),
            "spearman_supported_only": round(supported_spearman, 4),
            "spearman_controlled_context": round(controlled_spearman, 4),
            "question_control": summarize_binary(controls, "correct"),
            "all_user_turns": count_data["all_user_turns"],
            "checkpoint_proxy_user_turns": count_data["proxy_user_turns"],
            "selected_counts": selected_counts,
            "checkpoint_proxy_datasets": count_data["proxy_datasets"],
            "post_checkpoint_sources_excluded": count_data["excluded_late"],
        },
        "rows": rows,
        "controlled_context_rows": controlled_rows,
        "question_controls": controls,
    }


def parse_dialogues(path: Path) -> list[list[tuple[str, str]]]:
    dialogues = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for block in text.split("<|endoftext|>"):
        pairs = []
        pending = None
        for line in block.splitlines():
            if line.startswith("User:"):
                pending = line[len("User:"):].strip()
            elif line.startswith("Assistant:") and pending is not None:
                pairs.append((pending, line[len("Assistant:"):].strip()))
                pending = None
        if len(pairs) >= 2:
            dialogues.append(pairs)
    return dialogues


def heldout_candidates(path: Path, seed: int, per_type: int) -> list[dict]:
    dialogues = parse_dialogues(path)
    candidates = []
    for dialogue_index, pairs in enumerate(dialogues):
        for depth in range(1, min(len(pairs), 6)):
            question, reference = pairs[depth]
            candidates.append({
                "dialogue_index": dialogue_index,
                "depth": depth,
                "history": pairs[:depth],
                "question": question,
                "reference": reference,
                "question_type": "yes_no" if YES_NO_OPEN.search(question) else "open_or_statement",
            })
    random.Random(seed).shuffle(candidates)
    selected = []
    used_dialogues = defaultdict(set)
    for question_type in ("yes_no", "open_or_statement"):
        for candidate in candidates:
            if candidate["question_type"] != question_type:
                continue
            if candidate["dialogue_index"] in used_dialogues[question_type]:
                continue
            used_dialogues[question_type].add(candidate["dialogue_index"])
            selected.append(candidate)
            if len(used_dialogues[question_type]) >= per_type:
                break
    return selected


def run_claim_4(assistant: Assistant) -> dict:
    format_prompts = [
        "Reply with exactly two words that describe snow.",
        "Give only a number: how many days are in a leap year?",
        "Return exactly two bird names separated by a comma.",
        "Write exactly one sentence about a violin.",
        "Answer with a single adjective describing honey.",
        "Respond with only the city name: what is the capital of Spain?",
        "Use exactly three words to describe thunder.",
        "Output one vegetable name and nothing else.",
        "Put only the result of 3 plus 4 inside parentheses.",
        "Write a numbered list of four oceans.",
    ]
    original_max = config.MAX_NEW_TOKENS
    format_rows = []
    heldout_rows = []
    try:
        config.MAX_NEW_TOKENS = min(original_max, 24)
        for mode in ("chat", "auto"):
            for index, question in enumerate(format_prompts):
                row = ask(assistant, question, mode=mode, seed=40_000 + index)
                row["probe_mode"] = mode
                format_rows.append(row)

        corpora = {
            "soda": Path("data/soda/validation.txt"),
            "personachat": Path("data/personachat/validation.txt"),
        }
        for corpus_index, (corpus, path) in enumerate(corpora.items()):
            candidates = heldout_candidates(path, 41_000 + corpus_index, per_type=8)
            for index, candidate in enumerate(candidates):
                row = ask(
                    assistant,
                    candidate["question"],
                    history=candidate["history"],
                    mode="chat",
                    seed=42_000 + corpus_index * 100 + index,
                )
                row.update({
                    "corpus": corpus,
                    "question_type": candidate["question_type"],
                    "reference": candidate["reference"],
                    "reference_opens_with_verdict": opens_with_verdict(candidate["reference"]),
                    "dialogue_index": candidate["dialogue_index"],
                    "depth": candidate["depth"],
                })
                heldout_rows.append(row)
    finally:
        config.MAX_NEW_TOKENS = original_max

    format_summary = {}
    for mode in ("chat", "auto"):
        segment = [row for row in format_rows if row["probe_mode"] == mode]
        format_summary[mode] = {
            "opens_with_verdict": summarize_binary(segment, "opens_with_verdict"),
            "verdict_only": summarize_binary(segment, "verdict_only"),
            "retrieval_rate": round(rate(segment, "used_retrieval"), 4),
        }
    heldout_summary = {}
    for corpus in ("soda", "personachat"):
        for question_type in ("open_or_statement", "yes_no"):
            segment = [
                row for row in heldout_rows
                if row["corpus"] == corpus and row["question_type"] == question_type
            ]
            heldout_summary[f"{corpus}:{question_type}"] = {
                "generated_opens_with_verdict": summarize_binary(
                    segment, "opens_with_verdict"
                ),
                "reference_opens_with_verdict": summarize_binary(
                    segment, "reference_opens_with_verdict"
                ),
                "verdict_only": summarize_binary(segment, "verdict_only"),
            }
    return {
        "summary": {
            "format_probes": format_summary,
            "heldout_dialogues": heldout_summary,
        },
        "format_rows": format_rows,
        "heldout_rows": heldout_rows,
    }


SHORT_FILLERS = [
    ("I made tea this morning.", "That sounds pleasant."),
    ("The window is open.", "Fresh air can be nice."),
    ("I saw a red bicycle.", "Red bicycles are easy to notice."),
    ("Lunch was a sandwich.", "A sandwich is a practical lunch."),
    ("The street was quiet.", "Quiet streets can feel peaceful."),
    ("I watered a plant.", "The plant probably appreciated that."),
    ("A blue car passed by.", "Blue is a common car colour."),
    ("I read two pages.", "A little reading still counts."),
    ("The lamp is bright.", "Good lighting can be useful."),
    ("I heard a bird outside.", "Birdsong can be cheerful."),
    ("My cup is empty.", "It may be time for a refill."),
    ("The room feels warm.", "Opening a window might help."),
]

LONG_FILLERS = [
    (
        f"Here is unrelated observation number {index}: the ordinary room was quiet today.",
        "That sounds like a calm and completely ordinary observation with no special detail "
        "that needs to be remembered later.",
    )
    for index in range(1, 11)
]


def run_claim_5(assistant: Assistant) -> dict:
    facts = [
        ("code word", "Kestrel"),
        ("project nickname", "Orchid"),
        ("boat's name", "Mariner"),
        ("team mascot", "Badger"),
        ("folder label", "Cedar"),
        ("signal colour", "Amber"),
    ]
    rows = []
    for index, (key, expected) in enumerate(facts):
        fact_generic = (
            f"Please remember one detail: the {key} is {expected}.",
            "Okay, I will keep that detail in mind.",
        )
        fact_repeated = (
            fact_generic[0],
            f"Understood. The {key} is {expected}.",
        )
        base_question = f"What did I say the {key} was?"
        arrangements = [
            ("cap8_early_absent", ([fact_generic] + SHORT_FILLERS)[-8:], base_question),
            ("cap16_early_single", [fact_generic] + SHORT_FILLERS, base_question),
            ("cap16_early_repeated", [fact_repeated] + SHORT_FILLERS, base_question),
            (
                "cap16_structured_memory",
                [(
                    f"MEMORY RECORD: {key} = {expected}. Later, return this exact value.",
                    "Memory record acknowledged.",
                )] + SHORT_FILLERS,
                f"Memory test: return only the stored {key}.",
            ),
            ("cap16_early_long", [fact_generic] + LONG_FILLERS, base_question),
            (
                "cap16_early_initial_cue",
                [fact_generic] + SHORT_FILLERS,
                base_question + f" It began with the letter {expected[0]}.",
            ),
            ("cap16_recent", SHORT_FILLERS + [fact_generic], base_question),
            ("fact_only", [fact_generic], base_question),
            (
                "direct_current_turn",
                [],
                f"The {key} is {expected}. What is the {key}?",
            ),
        ]
        for arrangement_index, (arrangement, history, question) in enumerate(arrangements):
            row = ask(
                assistant,
                question,
                history=history,
                mode="chat",
                seed=50_000 + index,
            )
            lowered_prompt = row["prompt"].lower()
            target = expected.lower()
            first_at = lowered_prompt.find(target)
            last_at = lowered_prompt.rfind(target)
            first_after = None
            last_after = None
            if first_at >= 0:
                first_after = len(
                    assistant.generator.tokenizer.encode(row["prompt"][first_at:]).ids
                )
                last_after = len(
                    assistant.generator.tokenizer.encode(row["prompt"][last_at:]).ids
                )
            row.update({
                "arrangement": arrangement,
                "fact_key": key,
                "expected": expected,
                "expected_token_count": len(
                    assistant.generator.tokenizer.encode(expected).ids
                ),
                "correct": contains_expected(row["answer"], [expected]),
                "fact_in_prompt": first_at >= 0,
                "tokens_after_first_fact": first_after,
                "tokens_after_last_fact": last_after,
            })
            rows.append(row)

    summary = {}
    for arrangement in sorted({row["arrangement"] for row in rows}):
        segment = [row for row in rows if row["arrangement"] == arrangement]
        summary[arrangement] = {
            **summarize_binary(segment, "correct"),
            "fact_in_prompt": summarize_binary(segment, "fact_in_prompt"),
            "mean_prompt_tokens": round(mean(segment, "prompt_tokens"), 1),
            "mean_tokens_after_first_fact": round(
                mean(segment, "tokens_after_first_fact"), 1
            ),
            "mean_tokens_after_last_fact": round(
                mean(segment, "tokens_after_last_fact"), 1
            ),
        }
    by_fact = {}
    for key, expected in facts:
        segment = [row for row in rows if row["fact_key"] == key]
        by_fact[key] = {
            "expected": expected,
            "expected_token_count": segment[0]["expected_token_count"],
            **summarize_binary(segment, "correct"),
        }
    return {"summary": summary, "by_fact": by_fact, "rows": rows}


def render_text(result: dict) -> str:
    lines = [
        "INDEPENDENT CLAIMS VERIFICATION",
        f"elapsed: {result['run']['elapsed_seconds']:.1f}s",
        f"checkpoint: {result['run']['checkpoint']}",
        "",
        "CLAIM 1 - history dropped during retrieval",
    ]
    for name, summary in result["claim_1"]["summary"].items():
        lines.append(
            f"  {name:<34} {summary['hits']}/{summary['n']} "
            f"({summary['rate']:.1%}), support={summary['passage_support_rate']:.1%}"
        )
    lines.extend(["", "CLAIM 2 - all caps normalization"])
    for name, summary in result["claim_2"]["summary"].items():
        lines.append(f"  {name}: {summary}")
    lines.extend(["", "CLAIM 3 - opening verb frequency"])
    c3 = result["claim_3"]["summary"]
    lines.append(
        "  Spearman(log checkpoint-proxy count, accuracy): "
        f"{c3['spearman_log_frequency_vs_accuracy']:.3f}"
    )
    lines.append(
        f"  supported-only Spearman: {c3['spearman_supported_only']:.3f}; "
        f"fixed-context Spearman: {c3['spearman_controlled_context']:.3f}"
    )
    lines.append(f"  frequency bins: {c3['frequency_bins']}")
    lines.append(
        f"  fixed-context bins: {c3['controlled_context_frequency_bins']}"
    )
    lines.append(f"  question control: {c3['question_control']}")
    lines.append(f"  post-checkpoint source files: {c3['post_checkpoint_sources_excluded']}")
    for verb, summary in c3["by_verb"].items():
        lines.append(
            f"  {verb:<10} count={summary['checkpoint_proxy_count']:<6} "
            f"accuracy={summary['hits']}/{summary['n']} "
            f"support={summary['passage_support_rate']:.0%}"
        )
    lines.extend(["", "CLAIM 4 - verdict-word bias"])
    lines.append(
        "  format: " + json.dumps(result["claim_4"]["summary"]["format_probes"])
    )
    lines.append(
        "  held-out: " + json.dumps(result["claim_4"]["summary"]["heldout_dialogues"])
    )
    lines.extend(["", "CLAIM 5 - long-range recall"])
    for name, summary in result["claim_5"]["summary"].items():
        lines.append(
            f"  {name:<28} recall={summary['hits']}/{summary['n']} "
            f"fact-in-prompt={summary['fact_in_prompt']['hits']}/"
            f"{summary['fact_in_prompt']['n']} prompt_tokens={summary['mean_prompt_tokens']}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    started = time.time()
    original_drop = config.DROP_HISTORY_WHEN_RETRIEVING
    original_max = config.MAX_NEW_TOKENS
    original_normalizer = pipeline_module.normalise_shouting
    assistant = None
    try:
        print("loading assistant", flush=True)
        assistant = Assistant(device="cuda", quiet=True)
        result = {
            "run": {
                "checkpoint": assistant.generator.checkpoint,
                "tokenizer": str(assistant.generator.tokenizer_path),
                "context_length": assistant.generator.config.context_length,
                "repeats_claims_1_and_2": args.repeats,
            }
        }
        for label, function in (
            ("claim_1", lambda: run_claim_1(assistant, args.repeats)),
            ("claim_2", lambda: run_claim_2(assistant, args.repeats)),
            ("claim_3", lambda: run_claim_3(assistant)),
            ("claim_4", lambda: run_claim_4(assistant)),
            ("claim_5", lambda: run_claim_5(assistant)),
        ):
            print(f"running {label} ({time.time() - started:.1f}s)", flush=True)
            result[label] = function()
        result["run"]["elapsed_seconds"] = round(time.time() - started, 1)
        OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
        OUT_TEXT.write_text(render_text(result), encoding="utf-8")
        print(render_text(result), end="")
        print(f"wrote {OUT_JSON} and {OUT_TEXT}", flush=True)
        return 0
    finally:
        config.DROP_HISTORY_WHEN_RETRIEVING = original_drop
        config.MAX_NEW_TOKENS = original_max
        pipeline_module.normalise_shouting = original_normalizer
        if assistant is not None:
            assistant.close()


if __name__ == "__main__":
    sys.exit(main())
