"""Do the three conversation modes actually behave differently?

The knob added on 2026-08-19:

    auto   the router decides per message (default, and the behaviour before this)
    chat   never retrieve; the whole context budget goes to conversation history
    ask    always retrieve, except for questions about the conversation itself

Retrieval is not free. Context is 768 tokens, 96 reserved for the answer, so 672 remain - and
passages were measured eating 85 to 311 of them (up to 46%). Every token spent on a passage is a
token of conversation the model cannot see, which is why a mode that switches retrieval off buys
roughly double the retained conversation on a purely social exchange.

Checks:
  1. chat mode never retrieves, whatever the router says
  2. ask mode retrieves even on things the router calls chitchat
  3. ask mode still skips retrieval for personal questions, because Wikipedia cannot know them
  4. more history survives in chat mode than in ask mode
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from orchestrator import config
from orchestrator.pipeline import Assistant

FACTUAL = "What is the capital of France?"
SOCIAL = "hey, how's it going?"
PERSONAL = "what's my dog's name?"

passed = failed = 0


def check(label, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {label}  {detail}")
    else:
        failed += 1
        print(f"  FAIL  {label}  {detail}")


def main():
    assistant = Assistant(device="cuda", quiet=True)
    history = [("meet my dog, tommy", "Nice to meet Tommy!")]

    print("=== mode behaviour ===")
    for question in (FACTUAL, SOCIAL):
        chat = assistant.plan(question, has_history=True, mode="chat")
        ask = assistant.plan(question, has_history=True, mode="ask")
        auto = assistant.plan(question, has_history=True, mode="auto")
        check(f"chat never retrieves: {question[:32]!r}", not chat.used_retrieval)
        check(f"ask always retrieves: {question[:32]!r}", ask.used_retrieval,
              f"{len(ask.passages)} passages")
        print(f"        auto chose: {'retrieve' if auto.used_retrieval else 'no context'} "
              f"({auto.category} {auto.confidence:.2f})")

    print("\n=== personal questions stay out of retrieval in every mode ===")
    for mode in config.MODES:
        route = assistant.plan(PERSONAL, has_history=True, mode=mode)
        check(f"{mode:<5} does not retrieve a personal question", not route.used_retrieval)

    print("\n=== how much conversation survives ===")
    tokenizer = assistant.generator.tokenizer
    # Long enough turns that the budget actually binds. An earlier version used short turns that
    # both modes kept in full, so the test compared 12 against 12 and proved nothing - the
    # difference only appears once history is competing for space, which is the whole point.
    long_history = [
        (f"turn {i}: here is a fairly wordy question about a topic I care about, "
         f"with enough detail to take up realistic space in the prompt",
         f"turn {i}: and here is a correspondingly wordy answer that explains things "
         f"at the sort of length the model actually produces in conversation")
        for i in range(12)
    ]
    kept_by_mode = {}
    for mode in ("ask", "chat"):
        route = assistant.plan(FACTUAL, has_history=True, mode=mode)
        prompt, used = assistant.generator.build_prompt(FACTUAL, route.passages, long_history)
        # Each surviving turn contributes exactly one "User: turn N:" marker.
        kept = prompt.count("User: turn ")
        kept_by_mode[mode] = kept
        tokens = len(tokenizer.encode(prompt).ids)
        print(f"  {mode:<5} {used} passages | {kept:>2} of {len(long_history)} turns kept "
              f"| prompt {tokens} tokens")
    check("chat mode keeps more conversation than ask mode",
          kept_by_mode["chat"] > kept_by_mode["ask"],
          f"{kept_by_mode['chat']} vs {kept_by_mode['ask']} turns")

    print(f"\n{passed} passed, {failed} failed")
    assistant.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
