"""Replay Dark's looping conversation and check the loop is broken.

From `tests/dark_testing_results_pasted.txt`, chat mode collapsed into repeating one sentence:

    What's your name again?          -> Oh, Sam! I'm Sarah. Nice to meet you too.
    I thought you said Casey?        -> Oh, I thought you were Sarah. Nice to meet me too!
    I thought you're meeting me?     -> Oh, I'm Sarah. Nice to meet you too!
    Are you sarah or casey?          -> Yes, I am.
    yes, you are what?               -> Yes, I am.
    am what                          -> Oh, I'm Sarah. Nice to meet you too!
    oh ok. what's my name?           -> Sarah. Nice to meet you too!
    whhat's your name?               -> Oh, I'm Sarah. Nice to meet you too!
    how old are you sarah?           -> Sarah. Nice to meet you too!
    ...then "Asdf" for four turns, then "I'm sorry" for four turns.

The cause was structural, not random: the repetition penalty only covers tokens generated within
the CURRENT answer, and past answers live in the prompt, which the penalty deliberately skips.
Adding conversation history therefore created a feedback loop that could not exist before it -
each repeat entered history and made the next repeat likelier.

`ECHO_PENALTY` now applies to the model's own recent replies. This measures whether that helps,
by counting how many replies in a run are near-duplicates of the one before.
"""
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from orchestrator import config
from orchestrator.pipeline import Assistant

TURNS = [
    "Hi! My name is Sam",
    "What's your name again?",
    "I thought you just said your name is Casey?",
    "I thought you're meeting me? not you?",
    "Are you sarah or casey?",
    "yes, you are what?",
    "am what",
    "oh ok. what's my name?",
    "whhat's your name?",
    "how old are you?",
    "where do you wanna go for dinner?",
    "do you like music?",
]


def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# The state Dark's session actually reached: the same sentence already sitting in history several
# times over. Starting here is the honest test, because a fresh replay does not reliably fall into
# the attractor - the first version of this file ran the conversation from scratch, failed to
# reproduce the loop at all, and so compared two runs that were both fine. Seeding the degenerate
# state asks the question that matters: once stuck, can the model get out?
STUCK = "Oh, I'm Sarah. Nice to meet you too!"
SEEDED_HISTORY = [
    ("I thought you just said your name is Casey?", STUCK),
    ("I thought you're meeting me? not you?", STUCK),
    ("am what", STUCK),
    ("whhat's your name?", STUCK),
]
ESCAPE_TURNS = [
    "how old are you?",
    "where do you wanna go for dinner?",
    "do you like music?",
    "what did you do today?",
    "tell me something interesting",
    "are you ok?",
]


def escape_test(assistant, echo_penalty):
    original = config.ECHO_PENALTY
    config.ECHO_PENALTY = echo_penalty
    history = list(SEEDED_HISTORY)
    replies = []
    for turn in ESCAPE_TURNS:
        route = assistant.plan(turn, has_history=True, mode="chat")
        answer = assistant.answer(turn, route, history=history, mode="chat")
        replies.append(answer)
        if answer:
            history.append((turn, answer))
    config.ECHO_PENALTY = original
    stuck = sum(1 for r in replies if similar(r, STUCK) > 0.6)
    return replies, stuck


def run(assistant, echo_penalty):
    original = config.ECHO_PENALTY
    config.ECHO_PENALTY = echo_penalty
    history = []
    replies = []
    for turn in TURNS:
        route = assistant.plan(turn, has_history=bool(history), mode="chat")
        answer = assistant.answer(turn, route, history=history, mode="chat")
        replies.append(answer)
        if answer:
            history.append((turn, answer))
    config.ECHO_PENALTY = original

    loops = sum(1 for a, b in zip(replies, replies[1:]) if similar(a, b) > 0.7)
    unique = len({r.lower().strip() for r in replies})
    return replies, loops, unique


def main():
    assistant = Assistant(device="cuda", quiet=True)

    print("=" * 96)
    print("ESCAPE TEST - history is pre-loaded with the same reply four times over")
    print(f"stuck phrase: {STUCK!r}")
    print("=" * 96)
    for label, penalty in (("BEFORE (no echo penalty)", 1.0),
                           ("AFTER  (echo penalty on)", 1.15)):
        # Repeat the whole test so one unlucky roll cannot decide the answer.
        totals = []
        for attempt in range(3):
            replies, stuck = escape_test(assistant, penalty)
            totals.append(stuck)
            if attempt == 0:
                print(f"\n  {label}")
                for turn, reply in zip(ESCAPE_TURNS, replies):
                    flag = "  <- STILL STUCK" if similar(reply, STUCK) > 0.6 else ""
                    print(f"      {turn:<34} {reply[:60]}{flag}")
        print(f"      repeats of the stuck phrase across 3 runs: {totals} "
              f"(out of {len(ESCAPE_TURNS)} turns each)")

    assistant.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
