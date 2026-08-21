"""Broad stress test for DarkGLM's served behaviour.

This tool exists because one-off probes have repeatedly produced false conclusions in this repo.
It runs the real product path (router + retriever + model), logs route decisions and passages for
every sample, segments the results by behaviour, and writes raw JSONL evidence under `tests/`.

The scoring is deliberately simple and auditable: each probe has a small checker that marks whether
the answer satisfied the specific requirement being tested. The point is not to prove a benchmark
score; it is to find repeated failure modes with N>=5 evidence and enough routing/retrieval context
to diagnose whether the model, router, retriever, data mix, prompt format, or decoding is at fault.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from orchestrator.pipeline import Assistant


TEST_DIR = Path("tests")
DATA_DIR = Path("data")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def contains_any(text: str, needles: list[str]) -> bool:
    low = norm(text)
    return any(n in low for n in needles)


def token_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def sentence_count(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+", text.strip()) if s.strip()])


def numbered_count(text: str) -> int:
    return len(re.findall(r"(?:^|\s)(?:\d+[\).\:]|[-*])\s+", text))


def verdict_false_premise(answer: str) -> bool:
    return contains_any(answer, [
        "did not", "didn't", "was not", "wasn't", "not invent", "no evidence",
        "false", "incorrect", "wrong", "never", "i don't know", "unknown",
    ])


def asks_clarify(answer: str) -> bool:
    return contains_any(answer, [
        "clarify", "which", "what do you mean", "could mean", "ambiguous",
        "more information", "not sure", "i don't know", "please provide",
        "can you", "could you", "need", "unknown",
    ])


def one_word(expected: str | None = None) -> Callable[[str], bool]:
    def check(answer: str) -> bool:
        words = re.findall(r"[A-Za-z0-9]+", answer)
        if len(words) != 1:
            return False
        return expected is None or words[0].lower() == expected.lower()
    return check


def contains_expected(expected: str | list[str]) -> Callable[[str], bool]:
    needles = [expected] if isinstance(expected, str) else expected
    return lambda answer: contains_any(answer, [n.lower() for n in needles])


def number_only(expected: str) -> Callable[[str], bool]:
    return lambda answer: re.fullmatch(r"\s*" + re.escape(expected) + r"\s*\.?\s*", answer or "") is not None


def max_words(n: int) -> Callable[[str], bool]:
    return lambda answer: token_count(answer) <= n


def exact_list_count(n: int) -> Callable[[str], bool]:
    def check(answer: str) -> bool:
        # Accept either numbered/bulleted lists or comma-separated short lists.
        nums = numbered_count(answer)
        if nums:
            return nums == n
        parts = [p.strip() for p in re.split(r",|;|\n", answer) if p.strip()]
        return len(parts) == n and all(token_count(p) <= 5 for p in parts)
    return check


def one_sentence(answer: str) -> bool:
    return sentence_count(answer) == 1


PROBES: dict[str, list[dict]] = {
    "false_premise": [
        {"q": "When did Albert Einstein invent the telephone?", "check": verdict_false_premise},
        {"q": "Where was Marie Curie crowned queen of France?", "check": verdict_false_premise},
        {"q": "What year did Shakespeare write the Harry Potter books?", "check": verdict_false_premise},
        {"q": "Who was president of the United States in 1492?", "check": verdict_false_premise},
        {"q": "When did the Moon landing happen on Mars?", "check": verdict_false_premise},
        {"q": "What instrument did Beethoven use to paint the Mona Lisa?", "check": verdict_false_premise},
        {"q": "Where did Isaac Newton discover penicillin?", "check": verdict_false_premise},
        {"q": "How many Oscars did Mozart win for Star Wars?", "check": verdict_false_premise},
        {"q": "When was the Eiffel Tower built in ancient Egypt?", "check": verdict_false_premise},
        {"q": "Who invented gravity before Newton created it?", "check": verdict_false_premise},
    ],
    "comparison": [
        {"q": "Which is bigger, an elephant or a mouse?", "check": contains_expected("elephant")},
        {"q": "Which is larger, Jupiter or Earth?", "check": contains_expected("jupiter")},
        {"q": "Is Canada bigger than Luxembourg?", "check": contains_expected(["yes", "canada"])},
        {"q": "Which is smaller, a grape or a watermelon?", "check": contains_expected("grape")},
        {"q": "Which is heavier, a feather or a bowling ball?", "check": contains_expected("bowling")},
        {"q": "Is 9 bigger than 17?", "check": contains_expected(["no", "17"])},
        {"q": "Which is hotter, boiling water or ice?", "check": contains_expected("boiling")},
        {"q": "Which is longer, a year or a day?", "check": contains_expected("year")},
        {"q": "Which has more sides, a triangle or a square?", "check": contains_expected("square")},
        {"q": "Is a kilometer longer than a meter?", "check": contains_expected(["yes", "kilometer"])},
    ],
    "negation": [
        {"q": "Which of these is NOT a mammal: dog, whale, shark?", "check": contains_expected("shark")},
        {"q": "Which one is NOT a fruit: apple, banana, carrot?", "check": contains_expected("carrot")},
        {"q": "What is NOT a planet: Mars, Venus, the Moon?", "check": contains_expected("moon")},
        {"q": "Which is NOT a color: blue, seven, green?", "check": contains_expected("seven")},
        {"q": "Which animal does NOT fly: eagle, sparrow, elephant?", "check": contains_expected("elephant")},
        {"q": "Which is NOT a metal: iron, copper, wood?", "check": contains_expected("wood")},
        {"q": "Which is NOT a language: Spanish, piano, Arabic?", "check": contains_expected("piano")},
        {"q": "Which is NOT a season: winter, Tuesday, summer?", "check": contains_expected("tuesday")},
        {"q": "Which is NOT an ocean: Pacific, Sahara, Atlantic?", "check": contains_expected("sahara")},
        {"q": "Which is NOT used for breathing: lungs, gills, spoon?", "check": contains_expected("spoon")},
    ],
    "multi_hop": [
        {"q": "What language is mainly spoken in the country whose capital is Tokyo?", "check": contains_expected("japanese")},
        {"q": "What currency is used in the country whose capital is Paris?", "check": contains_expected("euro")},
        {"q": "What animal is the mascot of the company that makes the iPhone?", "check": contains_expected("apple")},
        {"q": "What color do you get by mixing the color of the sky with the color of a banana?", "check": contains_expected("green")},
        {"q": "What continent is the country that has Berlin as its capital in?", "check": contains_expected("europe")},
    ],
    "ambiguous": [
        {"q": "What is Java?", "check": asks_clarify},
        {"q": "Tell me about Mercury.", "check": asks_clarify},
        {"q": "What is Python used for?", "check": asks_clarify},
        {"q": "Who is Jordan?", "check": asks_clarify},
        {"q": "What happened in spring?", "check": asks_clarify},
    ],
    "format_constraints": [
        {"q": "Answer in one word: what color is grass?", "check": one_word("green")},
        {"q": "Answer with only the number: how many legs does a spider have?", "check": number_only("8")},
        {"q": "List exactly three primary colors, comma separated.", "check": exact_list_count(3)},
        {"q": "Write one sentence about photosynthesis.", "check": one_sentence},
        {"q": "Translate to Spanish, output only the translation: good morning", "check": lambda a: norm(a) in {"buenos días", "buenos dias"}},
        {"q": "Use exactly two words to describe the ocean.", "check": lambda a: token_count(a) == 2},
        {"q": "Give five animals as a numbered list and no explanation.", "check": exact_list_count(5)},
        {"q": "Reply with yes or no only: is water wet?", "check": lambda a: norm(a).strip(".") in {"yes", "no"}},
        {"q": "Summarise this in under 10 words: The cat slept on the warm windowsill all afternoon.", "check": max_words(10)},
        {"q": "Return only the chemical symbol for gold.", "check": lambda a: norm(a).strip(".") == "au"},
    ],
    "robustness_emptyish": [
        {"q": "", "check": asks_clarify},
        {"q": "   ", "check": asks_clarify},
        {"q": ".", "check": asks_clarify},
        {"q": "???", "check": asks_clarify},
        {"q": "!!!!", "check": asks_clarify},
        {"q": "asdf qwer zxcv", "check": asks_clarify},
        {"q": "blorple snorfle glax", "check": asks_clarify},
        {"q": "WHAT IS PHOTOSYNTHESIS?", "check": contains_expected(["plant", "light", "energy", "photosynthesis"])},
        {"q": "what is photosynthesis what is photosynthesis what is photosynthesis", "check": contains_expected(["plant", "light", "energy", "photosynthesis"])},
        {"q": "Hi" + " very" * 180 + ", what is a cat?", "check": contains_expected(["animal", "mammal", "cat"])},
    ],
    "consistency_pride": [
        {"q": "Who wrote Pride and Prejudice?", "check": contains_expected("jane austen"), "canon": "jane austen"},
        {"q": "Name the author of Pride and Prejudice.", "check": contains_expected("jane austen"), "canon": "jane austen"},
        {"q": "Pride and Prejudice was written by whom?", "check": contains_expected("jane austen"), "canon": "jane austen"},
        {"q": "Which novelist wrote Pride and Prejudice?", "check": contains_expected("jane austen"), "canon": "jane austen"},
        {"q": "Who is the writer behind Pride & Prejudice?", "check": contains_expected("jane austen"), "canon": "jane austen"},
        {"q": "What person authored the novel Pride and Prejudice?", "check": contains_expected("jane austen"), "canon": "jane austen"},
    ],
    "consistency_gold": [
        {"q": "What is the chemical symbol for gold?", "check": contains_expected("au"), "canon": "au"},
        {"q": "Gold has what chemical symbol?", "check": contains_expected("au"), "canon": "au"},
        {"q": "Which element symbol represents gold?", "check": contains_expected("au"), "canon": "au"},
        {"q": "Answer with the symbol for gold.", "check": contains_expected("au"), "canon": "au"},
        {"q": "What letters stand for gold on the periodic table?", "check": contains_expected("au"), "canon": "au"},
        {"q": "On the periodic table, gold is abbreviated as what?", "check": contains_expected("au"), "canon": "au"},
    ],
}


CONVERSATIONS = [
    {
        "name": "coreference_pet",
        "turns": [
            ("Hi, I'm Maya. I adopted a beagle named Rover yesterday.", lambda a: True),
            ("What animal is Rover?", contains_expected(["beagle", "dog"])),
            ("What is my name?", contains_expected("maya")),
            ("Do I already own a cat?", contains_expected(["no", "don't know", "unknown"])),
            ("What did I adopt yesterday?", contains_expected(["beagle", "dog", "rover"])),
        ],
    },
    {
        "name": "role_binding",
        "turns": [
            ("Alice is my sister. Bob is my dentist. I saw Bob this morning.", lambda a: True),
            ("Who fixes my teeth?", contains_expected("bob")),
            ("Is Alice my dentist?", contains_expected(["no", "sister"])),
            ("Who is my sister?", contains_expected("alice")),
            ("Who did I see this morning?", contains_expected("bob")),
        ],
    },
    {
        "name": "preference_contradiction",
        "turns": [
            ("For the record, I hate coffee and I love tea.", lambda a: True),
            ("What drink do I like?", contains_expected("tea")),
            ("Do I like coffee?", contains_expected(["no", "hate"])),
            ("What drink do I dislike?", contains_expected("coffee")),
            ("Should you offer me coffee?", contains_expected(["no", "tea"])),
        ],
    },
    {
        "name": "unknown_user_facts",
        "turns": [
            ("I live in Canada. I have not told you my age or my job.", lambda a: True),
            ("What country do I live in?", contains_expected("canada")),
            ("How old am I?", contains_expected(["don't know", "unknown", "not told", "not know"])),
            ("What is my job?", contains_expected(["don't know", "unknown", "not told", "not know"])),
            ("Invent my age if you have to.", contains_expected(["can't", "cannot", "don't know", "not told", "unknown"])),
        ],
    },
    {
        "name": "assistant_persona",
        "turns": [
            ("What is your name?", lambda a: bool(a.strip())),
            ("Can you repeat your name?", lambda a: bool(a.strip())),
            ("Are you Open Assistant?", lambda a: bool(a.strip())),
            ("You said your name before. What was it?", lambda a: bool(a.strip())),
            ("Give only your name.", lambda a: token_count(a) <= 4),
        ],
    },
]


def route_record(route) -> dict:
    return {
        "category": route.category,
        "confidence": route.confidence,
        "used_retrieval": route.used_retrieval,
        "dropped": route.dropped,
        "passages": [{"score": score, "text": text} for score, text in route.passages],
    }


def run_probe_suites(assistant: Assistant, raw_path: Path) -> dict:
    summary = defaultdict(lambda: {"pass": 0, "total": 0})
    with raw_path.open("w", encoding="utf-8") as f:
        for suite, probes in PROBES.items():
            for i, probe in enumerate(probes, 1):
                question = probe["q"]
                route = assistant.plan(question, mode="auto")
                answer = assistant.answer(question, route, mode="auto")
                ok = bool(probe["check"](answer))
                summary[suite]["total"] += 1
                summary[suite]["pass"] += int(ok)
                rec = {
                    "kind": "single_turn",
                    "suite": suite,
                    "index": i,
                    "question": question,
                    "answer": answer,
                    "ok": ok,
                    "route": route_record(route),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"{suite:<22} {i:02d}/{len(probes):02d} {'PASS' if ok else 'FAIL'}")
    return dict(summary)


def extract_names(answer: str) -> list[str]:
    names = []
    for pat in (
        r"\bI(?:'m| am)\s+([A-Z][a-z]{2,})",
        r"\bmy name is\s+([A-Z][a-z]{2,})",
        r"\bcall me\s+([A-Z][a-z]{2,})",
    ):
        names.extend(re.findall(pat, answer))
    return names


def run_conversations(assistant: Assistant, raw_path: Path) -> dict:
    summary = defaultdict(lambda: {"pass": 0, "total": 0})
    persona_names = []
    with raw_path.open("a", encoding="utf-8") as f:
        for convo in CONVERSATIONS:
            history = []
            convo_answers = []
            for i, (question, check) in enumerate(convo["turns"], 1):
                route = assistant.plan(question, has_history=bool(history), mode="chat")
                answer = assistant.answer(question, route, history=history, mode="chat")
                ok = bool(check(answer))
                summary[convo["name"]]["total"] += 1
                summary[convo["name"]]["pass"] += int(ok)
                if convo["name"] == "assistant_persona":
                    persona_names.extend(extract_names(answer))
                rec = {
                    "kind": "conversation",
                    "conversation": convo["name"],
                    "turn": i,
                    "question": question,
                    "answer": answer,
                    "ok": ok,
                    "history_before": history,
                    "route": route_record(route),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"{convo['name']:<22} {i:02d}/{len(convo['turns']):02d} {'PASS' if ok else 'FAIL'}")
                convo_answers.append(answer)
                history.append((question, answer))
            if convo["name"] == "assistant_persona":
                distinct = sorted(set(n.lower() for n in persona_names))
                summary["assistant_persona_name_stability"] = {
                    "pass": int(len(distinct) <= 1),
                    "total": 1,
                    "distinct_names": distinct,
                    "answers": convo_answers,
                }
    return dict(summary)


def iter_train_files() -> list[Path]:
    # Fine-tune behaviour is what this stress test diagnoses. Skip raw pretraining corpora:
    # Wikipedia/TinyStories/WikiText are huge and do not contain the masked assistant targets that
    # create the serving behaviours measured here.
    # This matches `tools.show_finetune_mix`: present-on-disk superseded experiments such as
    # `squad_v2`, raw `gsm8k`, `dailydialog`, `dolly`, and `strategyqa` are intentionally excluded.
    current_mix = {
        "commonsenseqa", "sciq", "piqa", "boolq", "hellaswag", "socialiqa", "logiqa",
        "reclor", "squad", "race", "xsum", "oasst1", "personachat", "proofwriter",
        "definitions", "yesno", "gsm8k_clean", "soda", "no_robots", "replay",
    }
    return sorted(
        p for p in DATA_DIR.glob("*/train.txt")
        if p.is_file() and p.parent.name in current_mix
    )


def audit_training_data(max_mb_per_file: int | None = None) -> dict:
    """Count surface evidence for likely roots without touching packed arrays."""
    prompt_patterns = {
        "format_constraint_prompt": re.compile(
            r"\b(one word|only the|answer with|exactly|comma[- ]separated|json|numbered list|"
            r"under \d+ words|at most \d+ words)\b", re.I
        ),
        "false_premise_prompt": re.compile(r"\b(false premise|trick question|did not invent|never existed|not true)\b", re.I),
        "ambiguous_prompt": re.compile(r"\b(ambiguous|which do you mean|clarify)\b", re.I),
        "comparison_prompt": re.compile(r"\b(bigger than|larger than|smaller than|heavier than|more sides)\b", re.I),
        "emptyish_prompt": re.compile(r"^\s*(?:[?.!]+|)\s*$"),
    }
    answer_patterns = {
        "unknown_answer": re.compile(r"\b(unknown|i don't know|not enough information|cannot answer|can't answer)\b", re.I),
        "assistant_name_claim": re.compile(r"\b(?:i'm|i am|my name is)\s+([A-Z][a-z]{2,})\b"),
        "open_assistant_claim": re.compile(r"\bOpen Assistant\b", re.I),
    }
    counts = Counter()
    name_counts = Counter()
    file_counts = {}
    for path in iter_train_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if max_mb_per_file is not None:
            text = text[: max_mb_per_file * 1024 * 1024]
        examples = [b.strip() for b in text.split("<|endoftext|>") if b.strip()]
        file_counts[str(path)] = len(examples)
        counts["examples"] += len(examples)
        for block in examples:
            counts["multi_turn_examples"] += int(block.count("\nUser:") > 1)
            first_user = block.split("\nAssistant:", 1)[0]
            last_answer = block.rsplit("\nAssistant:", 1)[-1] if "\nAssistant:" in block else ""
            prompt_text = re.sub(r"\nContext:.*", "", first_user, flags=re.S)
            for key, pat in prompt_patterns.items():
                counts[key] += int(bool(pat.search(prompt_text)))
            for key, pat in answer_patterns.items():
                match = pat.search(last_answer)
                counts[key] += int(bool(match))
                if key == "assistant_name_claim" and match:
                    name_counts[match.group(1)] += 1
    return {
        "counts": dict(counts),
        "top_assistant_name_claims": name_counts.most_common(20),
        "file_example_counts": file_counts,
    }


def write_summary(path: Path, single: dict, conv: dict, audit: dict) -> None:
    result = {
        "single_turn_summary": single,
        "conversation_summary": conv,
        "training_data_audit": audit,
    }
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--skip-model", action="store_true", help="Only run the training-data audit.")
    parser.add_argument("--audit-max-mb-per-file", type=int, default=None)
    args = parser.parse_args()

    stamp = args.out_prefix or datetime.now().strftime("darkglm_broad_stress_%Y%m%d_%H%M%S")
    TEST_DIR.mkdir(exist_ok=True)
    raw_path = TEST_DIR / f"{stamp}.jsonl"
    summary_path = TEST_DIR / f"{stamp}_summary.json"

    single_summary = {}
    conversation_summary = {}
    if not args.skip_model:
        assistant = Assistant(device=args.device, quiet=True)
        try:
            single_summary = run_probe_suites(assistant, raw_path)
            conversation_summary = run_conversations(assistant, raw_path)
        finally:
            assistant.close()

    audit = audit_training_data(args.audit_max_mb_per_file)
    write_summary(summary_path, single_summary, conversation_summary, audit)

    print("\nSUMMARY")
    for group, stats in {**single_summary, **conversation_summary}.items():
        if "pass" in stats and "total" in stats:
            print(f"  {group:<34} {stats['pass']:>3}/{stats['total']:<3} pass")
    print(f"\nRaw:     {raw_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
