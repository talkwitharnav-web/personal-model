"""Robustness: what happens with input the model was never shaped for?

Complements `tools/stress_qa.py` (facts, negation, format) and the conversation stress test.
This one is about INPUT SHAPE rather than content - the cases a real user produces by accident.

Each group has a specific suspicion behind it:

  empty-ish     punctuation only, a single letter, whitespace. Does it produce anything sane, or
                echo the input back? Echo has been this project's recurring failure mode.
  very long     a wall of text. The context is 768 tokens; what happens when the user turn alone
                approaches or exceeds it?
  caps / repeat ALL CAPS and repeated words. The tokenizer was fitted on clean Wikipedia, where
                sustained capitals are rare, so these may fragment badly.
  ambiguous     "it", "that one" with no referent. Should ask for clarification; probably will
                not, because only 1.1% of answers in the mix push back on anything.
  code / symbols non-prose input. Wikipedia has some, but the fine-tune mix is nearly all prose.

Reports the rate of three objective failures rather than judging quality:
  echo      the answer largely repeats the input
  empty     nothing came back
  degenerate the answer repeats itself internally
"""
import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from orchestrator.pipeline import Assistant

GROUPS = {
    "empty-ish": ["?", "...", "!!!", "a", "hm", "   .", "??", "-"],
    "caps and repetition": ["WHAT IS THE CAPITAL OF FRANCE",
                            "HELLO THERE",
                            "what what what is a dog",
                            "tell me tell me tell me about cats",
                            "AAAAAAAA"],
    "ambiguous reference": ["what is it?", "tell me about that one", "explain this",
                            "who is he?", "how does that work?"],
    "symbols and code": ["2 + 2 = ?", "def foo(): return 1", "{'a': 1}",
                         "http://example.com", "SELECT * FROM users"],
    "very long": ["Tell me about dogs. " * 60],
    "multilingual": ["¿Cuál es la capital de Francia?", "Was ist die Hauptstadt von Frankreich?",
                     "フランスの首都はどこですか"],
}


def repeats_itself(text, n=4):
    words = text.split()
    if len(words) < n * 2:
        return False
    seen = set()
    for i in range(len(words) - n + 1):
        gram = tuple(words[i:i + n])
        if gram in seen:
            return True
        seen.add(gram)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    # Accepted so tools/final_evaluation.py can drive this uniformly with the other evals. The
    # pipeline reads the checkpoint from orchestrator/config.py, so --checkpoint is recorded for
    # the report rather than used; --device is real.
    parser.add_argument("--checkpoint", default="checkpoints/finetune_v2_best.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assistant = Assistant(device=args.device, quiet=True)
    overall = {"echo": 0, "empty": 0, "degenerate": 0, "total": 0}

    for group, prompts in GROUPS.items():
        print(f"\n{'=' * 98}\n{group.upper()}\n{'=' * 98}")
        echo = empty = degenerate = total = 0
        for prompt in prompts:
            for _ in range(args.samples):
                route = assistant.plan(prompt, mode="auto")
                answer = assistant.answer(prompt, route, mode="auto")
                total += 1
                is_empty = not answer.strip()
                is_echo = (not is_empty
                           and SequenceMatcher(None, prompt.lower()[:200],
                                               answer.lower()[:200]).ratio() > 0.6)
                is_degen = repeats_itself(answer)
                empty += is_empty
                echo += is_echo
                degenerate += is_degen
            flags = []
            if is_empty:
                flags.append("EMPTY")
            if is_echo:
                flags.append("ECHO")
            if is_degen:
                flags.append("LOOP")
            shown = prompt if len(prompt) <= 46 else prompt[:43] + "..."
            tag = "rag" if route.used_retrieval else "   "
            print(f"  [{tag}] {shown:<48} {answer[:52]!r}"
                  + (f"  <- {'/'.join(flags)}" if flags else ""))
        print(f"    -> echo {echo}/{total}  empty {empty}/{total}  degenerate {degenerate}/{total}")
        for key, value in (("echo", echo), ("empty", empty), ("degenerate", degenerate),
                           ("total", total)):
            overall[key] += value

    n = overall["total"]
    print(f"\n{'=' * 98}\nOVERALL over {n} generations")
    print(f"  echo       {overall['echo']:>3}  ({100*overall['echo']/n:.0f}%)")
    print(f"  empty      {overall['empty']:>3}  ({100*overall['empty']/n:.0f}%)")
    print(f"  degenerate {overall['degenerate']:>3}  ({100*overall['degenerate']/n:.0f}%)")
    assistant.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
