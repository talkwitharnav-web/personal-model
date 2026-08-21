"""Prove conversation history reaches the model - using Dark's own failing example.

The bug he hit:

    user: meet my dog, tommy
    assistant: Thanks, I'm glad you're enjoying your day
    user: what's my dog's name?
    assistant: Yes                     <- retrieved random Wikipedia dog-name passages

`chat.py` sent every message as turn 1 of a brand new conversation, so "what's my dog's name?"
arrived with no mention of a dog ever having been introduced. The model is capable of this - it
tracked "he" back to a dog named Rex across three turns when the turns were actually supplied -
it simply was never given them.

This runs the same exchange twice, without history and with, so the difference is visible rather
than asserted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from orchestrator.pipeline import Assistant

CONVERSATIONS = [
    ["meet my dog, tommy", "what's my dog's name?"],
    ["my name is Dark", "what is my name?"],
    ["I have a cat called Luna", "what colour is she?", "is she friendly?"],
    ["I'm learning to play guitar", "how long does it usually take?"],
]


def run(assistant, turns, use_history):
    history = []
    label = "WITH history" if use_history else "NO history (the old behaviour)"
    print(f"\n  --- {label} ---")
    for turn in turns:
        # has_history is what enables the personal-context rule, so it has to be passed the same
        # way chat.py passes it - an earlier version of this test omitted it and reported the
        # rule as not firing when it had simply never been switched on.
        route = assistant.plan(turn, has_history=bool(history) and use_history)
        answer = assistant.answer(turn, route, history=history if use_history else ())
        tag = "rag" if route.used_retrieval else "no ctx"
        print(f"      user      {turn}")
        print(f"      assistant [{tag}] {answer[:96]}")
        if answer:
            history.append((turn, answer))


def main():
    assistant = Assistant(device="cuda", quiet=True)
    for turns in CONVERSATIONS:
        print("=" * 96)
        run(assistant, turns, use_history=False)
        run(assistant, turns, use_history=True)
    assistant.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
