"""Look for SHORTCUTS across every dataset in the v2 fine-tune mix.

A shortcut is any way to get the right answer without doing the task. Section 9 is the record of
what they cost this project: ProofWriter answerable by counting statements, SQuAD v2 answerable by
always refusing, definitions answerable by echoing the subject. Every one of them produced a good
validation loss and a useless model.

The yes/no set just supplied another: EVERY example carries a `Context:` block that contains the
answer, so the model is never asked to answer from its own knowledge. It scores well on the task it
was trained on and cannot answer a bare question at all.

Checks implemented, all cheap and all textual:

  majority        the most common answer's share. High = guess the majority and win.
  answer-in-ctx   how often the answer string appears verbatim in the context. High is CORRECT for
                  extractive QA (SQuAD) and a SHORTCUT everywhere else, because the model learns to
                  copy rather than to know.
  ctx-always      what fraction of examples have a context at all. 100% means the model never
                  practises answering unaided - it will fail every bare question at serving time.
  length-tell     for short-answer sets, correlation between answer length and its being correct is
                  not measurable without distractors, so this reports the answer-length spread
                  instead: a near-constant length is itself a template smell.
  dupe-answers    share of answers that are duplicates. High means a tiny answer vocabulary.
  q-dupes         share of duplicated questions - inflates apparent size without adding signal.

Read the numbers, not the verdicts: the point is to find things worth looking at by eye.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MIX = ["commonsenseqa", "sciq", "piqa", "boolq", "hellaswag", "socialiqa",
       "logiqa", "reclor", "squad", "race", "xsum", "oasst1", "personachat",
       "proofwriter", "definitions", "yesno", "gsm8k_clean", "soda", "no_robots", "replay",
       "typo_aug"]

# Extractive QA is SUPPOSED to have the answer in the context - that is the task, not a shortcut.
# SciQ ships a human-written "support" paragraph for the same reason SQuAD ships a passage, and
# BoolQ is a reading-comprehension benchmark whose whole point is judging a claim against one.
EXTRACTIVE = {"squad", "race", "reclor", "logiqa", "xsum", "sciq", "boolq"}

# Sets whose answer space is genuinely binary. A 50/50 split is the BEST possible balance for
# these, so the majority-class check has to be read against 50%, not against 40%. Flagging
# perfectly balanced yes/no data as "majority shortcut" is the averaging-different-populations
# mistake in another costume: the threshold only means something within one kind of task.
BINARY = {"boolq", "yesno"}

SEPARATOR = "<|endoftext|>"


def parse(text, limit):
    """Yield (question, context, answer) from the User/Context/Assistant format."""
    out = []
    for block in text.split(SEPARATOR):
        block = block.strip()
        if not block:
            continue
        match = re.search(r"\nAssistant:", block)
        if not match:
            continue
        head, answer = block[:match.start()], block[match.end():].strip()
        context = ""
        ctx_match = re.search(r"\n\s*Context:\s*", head)
        if ctx_match:
            context = head[ctx_match.end():].strip()
            head = head[:ctx_match.start()]
        question = re.sub(r"^User:\s*", "", head).strip()
        out.append((question, context, answer))
        if len(out) >= limit:
            break
    return out


def audit(name, rows):
    answers = [a for _, _, a in rows]
    questions = [q for q, _, _ in rows]
    total = len(rows)
    if not total:
        return None

    counts = Counter(answers)
    majority = counts.most_common(1)[0][1] / total

    with_context = sum(1 for _, c, _ in rows if c)
    ctx_share = with_context / total

    leaked = 0
    for _, context, answer in rows:
        if not context or not answer:
            continue
        probe = answer.strip().strip(".").lower()
        if len(probe) >= 3 and probe in context.lower():
            leaked += 1
    leak = leaked / max(with_context, 1)

    lengths = [len(a.split()) for a in answers]
    mean_len = sum(lengths) / total
    spread = (max(lengths) - min(lengths)) if lengths else 0

    dupe_answers = 1 - len(set(answers)) / total
    dupe_questions = 1 - len(set(questions)) / total

    return {"n": total, "majority": majority, "ctx": ctx_share, "leak": leak,
            "mean_len": mean_len, "spread": spread,
            "dupe_a": dupe_answers, "dupe_q": dupe_questions,
            "top": counts.most_common(3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20000,
                        help="examples to read per dataset (streaming head, not the whole file)")
    args = parser.parse_args()

    print(f"reading up to {args.limit:,} examples per dataset\n")
    header = (f"{'dataset':<15}{'n':>8}{'major':>8}{'ctx':>7}{'ans-in-ctx':>12}"
              f"{'ans len':>9}{'dupA':>7}{'dupQ':>7}")
    print(header)
    print("-" * len(header))

    flagged = []
    for name in MIX:
        path = Path("data") / name / "train.txt"
        if not path.exists():
            print(f"{name:<15}  MISSING {path}")
            continue
        # Stream a head rather than loading 155 MB files whole (section 2: memory discipline).
        chunk = []
        size = 0
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                chunk.append(line)
                size += len(line)
                if size > 40_000_000:
                    break
        rows = parse("".join(chunk), args.limit)
        stats = audit(name, rows)
        if not stats:
            print(f"{name:<15}  could not parse")
            continue

        leak_note = f"{stats['leak']*100:>10.1f}%" if stats["ctx"] > 0 else f"{'-':>11}"
        print(f"{name:<15}{stats['n']:>8,}{stats['majority']*100:>7.1f}%"
              f"{stats['ctx']*100:>6.0f}%{leak_note}"
              f"{stats['mean_len']:>9.1f}{stats['dupe_a']*100:>6.0f}%{stats['dupe_q']*100:>6.0f}%")

        limit = 0.60 if name in BINARY else 0.40
        if stats["majority"] > limit:
            flagged.append((name, f"majority answer is {stats['majority']*100:.1f}% of the set"))
        if stats["ctx"] > 0.99 and name not in EXTRACTIVE:
            flagged.append((name, "EVERY example has a context - never practises answering unaided"))
        if stats["leak"] > 0.90 and name not in EXTRACTIVE:
            flagged.append((name, f"answer is copyable from the context in "
                                  f"{stats['leak']*100:.0f}% of examples"))
        if stats["dupe_q"] > 0.20:
            flagged.append((name, f"{stats['dupe_q']*100:.0f}% of questions are duplicates"))

    print("\n" + "=" * 78 + "\nFLAGGED\n" + "=" * 78)
    if not flagged:
        print("  nothing tripped a threshold")
    for name, why in flagged:
        print(f"  {name:<15} {why}")
    print("\nNote: high answer-in-context is CORRECT for extractive sets "
          f"({', '.join(sorted(EXTRACTIVE))}) and is not flagged for them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
