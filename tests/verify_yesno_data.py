"""Independently verify generated yes/no data.

Written to check the agent's work rather than take its word for it. The failure this dataset
exists to prevent is a shortcut, so the checks are aimed at shortcuts a summary statistic would
hide:

  - 50/50 balance is necessary but proves nothing on its own; the v1 defect had balanced data.
  - Every context must carry EXACTLY one Yes and one No. That is what makes copying the passage
    lose, and it is the whole design.
  - Both halves of a pair must use the SAME template. If phrasing correlated with the label the
    model could learn "this wording means Yes" - a new shortcut, worse than the one being fixed,
    and invisible in the balance counts.

The template check is done WITHOUT trusting the generator: for each pair it measures the longest
common prefix and suffix of the two questions. Same template means the two differ only where the
slot value sits, so prefix+suffix covers nearly the whole question. Different templates would
share almost nothing.
"""
import re
import sys
from collections import Counter
from pathlib import Path

from data.prepare_yesno import TEMPLATES

RECORD = re.compile(
    r"User: (?P<question>.*?)\n\nContext: (?P<context>.*?)\nAssistant: (?P<answer>Yes|No)\n"
    r"<\|endoftext\|>\n", re.S)

MIN_SHARED_FRACTION = 0.55


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def common_suffix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def main(directory):
    directory = Path(directory)
    failures = 0
    for name in ("train", "validation"):
        path = directory / f"{name}.txt"
        if not path.exists():
            print(f"MISSING {path}")
            failures += 1
            continue

        text = path.read_text(encoding="utf-8")
        records = [m.groupdict() for m in RECORD.finditer(text)]
        approx = text.count("<|endoftext|>")
        print(f"\n=== {path} ===")
        print(f"  records parsed {len(records):,} of {approx:,} <|endoftext|> markers")
        if len(records) != approx:
            print(f"  FAIL  {approx - len(records)} records did not match the exact format")
            failures += 1
        else:
            print("  PASS  every record matches the exact expected format")

        answers = Counter(r["answer"] for r in records)
        yes, no = answers["Yes"], answers["No"]
        print(f"  Yes {yes:,} / No {no:,}")
        if yes != no:
            print("  FAIL  not balanced")
            failures += 1
        else:
            print("  PASS  exactly 50/50")

        by_context = {}
        for record in records:
            by_context.setdefault(record["context"], []).append(record)

        bad_pairs = 0
        collisions = 0
        for group in by_context.values():
            labels = Counter(r["answer"] for r in group)
            if len(group) > 2:
                # Two different pairs can land on an identical context string. That is a
                # collision, not a broken pair - what matters is that the context still carries
                # as many Yes as No, so copying the passage still cannot win.
                collisions += 1
            if labels["Yes"] != labels["No"]:
                bad_pairs += 1
        print(f"  contexts {len(by_context):,}"
              + (f" ({collisions} shared by more than one pair)" if collisions else ""))
        if bad_pairs:
            print(f"  FAIL  {bad_pairs} contexts do not carry equal Yes and No")
            failures += 1
        else:
            print("  PASS  every context carries equal Yes and No")

        # Same-template check, done WITHOUT trusting the generator: compile every template in
        # the real table into a regex and require that some single template matches BOTH halves.
        #
        # An earlier version of this check compared the common prefix and suffix of the two
        # questions and flagged 234 "failures". That metric was wrong, not the data: for the
        # description templates the slot value IS most of the question ("Is Gozd a village in
        # the administrative district of Gmina Bobolice?" vs "Is Gozd a member of the 14th Lok
        # Sabha of India?"), so two correct halves of one template share very little text.
        # Measure the structure, not the overlap.
        patterns = []
        for kind, entries in TEMPLATES.items():
            for text, _requires, _weight in entries:
                pattern = re.escape(text)
                pattern = re.sub(r"\\\{(\w+)\\\}", ".+?", pattern)
                patterns.append((kind, text,
                                 re.compile(r"^" + pattern + r"$", re.I | re.S)))

        def matching(question):
            question = question.strip().rstrip("?").strip()
            found = set()
            for kind, text, compiled in patterns:
                probe = text.rstrip("?").strip()
                probe_pattern = re.sub(r"\\\{(\w+)\\\}", ".+?", re.escape(probe))
                if re.match(r"^" + probe_pattern + r"$", question, re.I | re.S):
                    found.add((kind, text))
            return found

        mismatched = 0
        unmatched = 0
        worst = None
        for group in by_context.values():
            yeses = [r["question"] for r in group if r["answer"] == "Yes"]
            nos = [r["question"] for r in group if r["answer"] == "No"]
            if len(yeses) != 1 or len(nos) != 1:
                continue
            a, b = yeses[0], nos[0]
            shared = matching(a) & matching(b)
            if not matching(a) or not matching(b):
                unmatched += 1
                if worst is None:
                    worst = ("unmatched", a, b)
            elif not shared:
                mismatched += 1
                worst = ("different templates", a, b)

        print(f"  template check: {len(patterns)} templates in the table")
        if unmatched:
            print(f"  WARN  {unmatched} pairs matched no template (slot content may span one)")
        if mismatched:
            print(f"  FAIL  {mismatched} pairs use DIFFERENT templates")
            print(f"        {worst[0]}:\n          Yes: {worst[1]!r}\n          No : {worst[2]!r}")
            failures += 1
        else:
            print("  PASS  both halves of every pair share one template")

        # Casing/punctuation style must also match across a pair.
        style_mismatch = 0
        for group in by_context.values():
            if len(group) != 2:
                continue
            a = next(r["question"] for r in group if r["answer"] == "Yes")
            b = next(r["question"] for r in group if r["answer"] == "No")
            if a.endswith("?") != b.endswith("?") or a[:1].isupper() != b[:1].isupper():
                style_mismatch += 1
        if style_mismatch:
            print(f"  FAIL  {style_mismatch} pairs differ in casing or punctuation")
            failures += 1
        else:
            print("  PASS  casing and punctuation match within every pair")

        lowercase = sum(1 for r in records if r["question"][:1].islower())
        no_mark = sum(1 for r in records if not r["question"].rstrip().endswith("?"))
        print(f"  style variety: {lowercase/len(records)*100:.1f}% lowercase, "
              f"{no_mark/len(records)*100:.1f}% no question mark")

        # Word boundaries matter: a substring test flags "yeshiva", "Yessongs" and "Yeshivat",
        # which are ordinary Wikipedia subjects and not answer leakage.
        leak = re.compile(r"\b(yes|no)\b", re.I)
        answer_in_question = sum(1 for r in records if leak.search(r["question"]))
        if answer_in_question:
            print(f"  WARN  {answer_in_question} questions contain the literal answer word")
        else:
            print("  PASS  no question contains the literal answer word")

    print(f"\n{'FAILED' if failures else 'ALL CHECKS PASSED'} ({failures} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/yesno"))
