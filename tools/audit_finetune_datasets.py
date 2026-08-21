"""Audit every fine-tune dataset for garbage AND for learnable shortcuts.

The central lesson of this project (onboarding section 9): the model minimises loss, so if a cheap
surface rule fits most of the data it learns that instead of the thing you meant. Nearly every
quality problem here was a shortcut.

Crucially, a data-distribution audit ALONE WOULD HAVE MISSED the worst one. BoolQ was balanced
50.6/49.4 and the model still ignored the question entirely, because every BoolQ passage appears
with exactly one question, so "echo the passage's polarity" scores 100%. Counting labels is not
enough - you have to ask whether a rule that never reads the question can win.

So this reports two things per dataset:
  GARBAGE  - empty answers, duplicates, degenerate answer distributions, answer leaked into the
             question, truncation artifacts, encoding damage
  SHORTCUTS- concrete cheap rules, scored. If a rule scores far above chance, the model will
             find it.

Rules probed:
  copy-first-sentence   answer is (a substring of) the context's opening sentence
  copy-any-context      answer appears verbatim anywhere in the context
  answer-in-question    answer appears in the question itself
  majority-class        always guess the most common answer
  one-question-per-ctx  does each context appear with exactly one question? (the BoolQ defect -
                        if so, "echo the passage" is never punished)
  position bias         for multiple choice, is the right option always in one place?
"""
import re
import sys
from collections import Counter
from pathlib import Path

MARKER = "\nAssistant:"
CONTEXT_SEPARATORS = ("\n\nContext:", "\nContext:", "\n\nOptions:", "\nOptions:")

DATA = Path("data")
SKIP = {"pretrain", "combined", "__pycache__", "wikipedia", "wikitext", "tinystories"}
SAMPLE = 6000


def parse_record(chunk):
    """Split on the FIRST '\\nAssistant:', which is exactly what the tokenize scripts do.

    The datasets do not share a layout - some use 'User: q\\n\\nContext: ...', some
    'User: q\\n\\nOptions:\\n- a\\n- b', and RACE puts the whole passage inside the User block
    with no marker at all. Keying off '\\nAssistant:' is the one thing they all agree on, and it
    matches where loss masking starts, so 'answer' here is the region that actually carries
    gradient.
    """
    chunk = chunk.strip()
    if not chunk.startswith("User:"):
        return None
    index = chunk.find(MARKER)
    if index == -1:
        return None
    prompt = chunk[:index]
    answer = chunk[index + len(MARKER):].strip()

    context = ""
    question = prompt
    for separator in CONTEXT_SEPARATORS:
        if separator in prompt:
            question, _, context = prompt.partition(separator)
            break
    else:
        # No explicit Context:/Options: block. RACE and ProofWriter carry the passage inline and
        # put the question on the LAST line, so splitting there recovers the real context.
        # Treating the whole prompt as the context makes every row unique and silently defeats
        # the context-reuse check - it reported 0.2% reuse for a ProofWriter set that genuinely
        # has 2.08 questions per theory.
        head, newline, tail = prompt.rpartition("\n")
        if newline and head.strip():
            context, question = head, tail
        else:
            context, question = "", prompt

    question = question[len("User:"):].strip() if question.startswith("User:") else question.strip()
    context = context[len("User:"):].strip() if context.startswith("User:") else context.strip()
    return {"question": question, "context": context or prompt.strip(), "answer": answer}


def read_records(path, limit):
    """Split on the separator first, then parse each record.

    Running one non-greedy regex across a whole file is quadratic-ish on the big ones - xsum is
    466 MB and the audit sat on it for 17 minutes without finishing. Splitting first bounds the
    work to a single record, and reading incrementally means the 466 MB file is never held in
    memory at once.
    """
    records = []
    buffer = ""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        while len(records) < limit:
            data = handle.read(8_000_000)
            if not data:
                break
            buffer += data
            parts = buffer.split("<|endoftext|>")
            buffer = parts.pop()
            for part in parts:
                record = parse_record(part)
                if record:
                    records.append(record)
                if len(records) >= limit:
                    break
    return records


def count_records(path):
    """Total record count without holding the file."""
    total = 0
    with open(path, "rb") as handle:
        while True:
            data = handle.read(16_000_000)
            if not data:
                break
            total += data.count(b"<|endoftext|>")
    return total


def context_reuse(path):
    """How many contexts appear with more than one question, across the WHOLE file.

    Measuring this on a 6,000-row prefix is wrong for any shuffled dataset: the regrouped
    ProofWriter genuinely has 2.08 questions per theory, but its rows are shuffled globally, so
    a prefix almost never contains two rows from the same theory. The audit reported 0.0% reuse
    and fired the BoolQ-defect warning at a dataset that had just been fixed.

    Streams the file and keeps only 8-byte hashes, so 200k contexts cost ~2 MB.
    """
    seen = set()
    repeated = set()
    buffer = ""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        while True:
            data = handle.read(8_000_000)
            if not data:
                break
            buffer += data
            parts = buffer.split("<|endoftext|>")
            buffer = parts.pop()
            for part in parts:
                record = parse_record(part)
                if not record or not record["context"]:
                    continue
                key = hash(record["context"])
                if key in seen:
                    repeated.add(key)
                else:
                    seen.add(key)
    return len(repeated), len(seen)


def first_sentence(text):
    parts = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)
    return parts[0] if parts else text


def audit(path, name):
    records = read_records(path, SAMPLE)
    if not records:
        print(f"\n### {name}: no parseable records ({path})")
        return None
    total = count_records(path)

    answers = [(r["answer"] or "").strip() for r in records]
    questions = [(r["question"] or "").strip() for r in records]
    contexts = [(r["context"] or "").strip() for r in records]

    print(f"\n### {name}  ({total:,} records, auditing {len(records):,})")

    # ---------- garbage ----------
    empty = sum(1 for a in answers if not a)
    counts = Counter(answers)
    top_answer, top_count = counts.most_common(1)[0]
    duplicate_questions = len(questions) - len(set(questions))
    replacement = sum(1 for a in answers if "\ufffd" in a)
    truncated = sum(1 for a in answers if a.endswith((" the", " a", " of", " in", " and", " to")))
    lengths = sorted(len(a) for a in answers)

    print(f"  answers: empty {empty} | unique {len(counts):,} | "
          f"median len {lengths[len(lengths)//2]} | p90 {lengths[len(lengths)*9//10]}")
    print(f"  most common answer {top_answer[:40]!r} = {top_count/len(records)*100:.1f}%")
    if duplicate_questions:
        print(f"  duplicate questions: {duplicate_questions:,} "
              f"({duplicate_questions/len(records)*100:.1f}%)")
    if replacement:
        print(f"  GARBAGE encoding damage in {replacement} answers")
    if truncated:
        print(f"  GARBAGE {truncated} answers end on a dangling function word")
    if empty:
        print(f"  GARBAGE {empty} empty answers")

    # ---------- shortcuts ----------
    with_context = [(q, c, a) for q, c, a in zip(questions, contexts, answers) if c and a]
    print(f"  -- shortcut probes ({len(with_context):,} rows with a context) --")
    print(f"  majority-class            {top_count/len(records)*100:5.1f}%")

    if with_context:
        copy_first = sum(1 for _, c, a in with_context
                         if a.lower() in first_sentence(c).lower())
        copy_any = sum(1 for _, c, a in with_context if a.lower() in c.lower())
        in_question = sum(1 for q, _, a in with_context if a.lower() in q.lower())
        n = len(with_context)
        print(f"  copy-first-sentence       {copy_first/n*100:5.1f}%")
        print(f"  copy-anywhere-in-context  {copy_any/n*100:5.1f}%")
        print(f"  answer-inside-question    {in_question/n*100:5.1f}%"
              + ("   <-- LEAK" if in_question / n > 0.5 else ""))

        # Measured over the WHOLE file, not the sample - see context_reuse().
        reused, unique_contexts = context_reuse(path)
        share = reused / max(1, unique_contexts)
        print(f"  contexts reused by >1 question (whole file): "
              f"{reused:,}/{unique_contexts:,} ({share*100:.1f}%)")
        # Only meaningful where the answer is a LABEL that could be echoed. For summarisation or
        # definition generation, one question per context is inherent to the task and flagging it
        # is crying wolf - the BoolQ defect was specifically "echo the passage's polarity".
        classification_like = len(counts) <= 10
        if classification_like and share < 0.05:
            print("    ^ WARNING: nearly every context has exactly ONE question, and the answer "
                  "is one of a few labels.")
            print("      'Echo the passage' is never punished here - this is the BoolQ defect.")

    if len(counts) <= 10:
        spread = ", ".join(f"{a[:18]!r} {c/len(records)*100:.1f}%"
                           for a, c in counts.most_common())
        print(f"  class distribution: {spread}")
        if top_count / len(records) > 1.5 / len(counts):
            print(f"    ^ skewed: guessing the majority beats even odds "
                  f"({top_count/len(records)*100:.1f}% vs {100/len(counts):.1f}%)")

    # multiple-choice position bias
    options = [q for q in questions if re.search(r"\bA[).]\s", q) and re.search(r"\bB[).]\s", q)]
    if len(options) > 20:
        letters = Counter(a.strip()[:1].upper() for a in answers if a.strip()[:1].upper()
                          in "ABCDE")
        if letters:
            spread = ", ".join(f"{k} {v/sum(letters.values())*100:.0f}%"
                               for k, v in sorted(letters.items()))
            print(f"  multiple-choice answer position: {spread}")

    return {"name": name, "total": total, "majority": top_count / len(records)}


def main():
    print("=" * 78)
    print("FINE-TUNE DATASET AUDIT - garbage and shortcuts")
    print("=" * 78)
    results = []
    for directory in sorted(DATA.iterdir()):
        if not directory.is_dir() or directory.name in SKIP:
            continue
        train = directory / "train.txt"
        if not train.exists():
            continue
        result = audit(train, directory.name)
        if result:
            results.append(result)

    print("\n" + "=" * 78)
    print("SUMMARY - datasets where guessing the majority class already scores high")
    print("=" * 78)
    for result in sorted(results, key=lambda r: -r["majority"]):
        flag = "  <-- imbalanced" if result["majority"] > 0.6 else ""
        print(f"  {result['name']:<18} {result['total']:>8,} rows | "
              f"majority {result['majority']*100:5.1f}%{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
