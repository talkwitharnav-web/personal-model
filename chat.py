import argparse
import sys

from orchestrator import config
from orchestrator.generation import pick_device
from orchestrator.pipeline import Assistant

# How many past turns to keep. The token budget in build_prompt trims to fit anyway - passages get
# priority, then history newest-first - so this is a cheap upper bound, not the real limit.
#
# Raised from 8 after measurement: at 8 turns a fact stated in turn 1 was gone from the prompt by
# turn 15 (fact-in-prompt 0/5). At 16 it is present 80% of the time and still fits comfortably -
# 581 of the 672 usable tokens. Note this alone does NOT fix recall: with the fact provably in the
# prompt, recall stayed 0.0, which is a capability limit rather than a truncation bug. It is still
# strictly better to let the budget decide than to throw turns away early.
MAX_HISTORY_TURNS = 16


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with DarkGLM.")
    parser.add_argument("question", nargs="*", help="ask once and exit")
    parser.add_argument("--cpu", action="store_true",
                        help="force CPU (use this while something is training)")
    parser.add_argument("--quiet", action="store_true",
                        help="hide routing and retrieved passages")
    parser.add_argument("--mode", choices=config.MODES, default=config.DEFAULT_MODE,
                        help="auto: the router decides (default). chat: never retrieve, spend "
                             "the whole context on conversation. ask: always look things up.")
    return parser.parse_args()


def show_route(route):
    note = f"  ({route.note})" if route.note else ""
    print(f"[{route.mode}] {route.category} {route.confidence:.2f}{note}")
    if not route.used_retrieval:
        return
    print(f"[rag] dropped {route.dropped} boilerplate candidates")
    for rank, (score, passage) in enumerate(route.passages, start=1):
        preview = passage if len(passage) <= 150 else passage[:150] + "..."
        print(f"  [{rank}] {score:.3f}  {preview}")


def respond(assistant, question, quiet, history, mode):
    route = assistant.plan(question, has_history=bool(history), mode=mode)
    if not quiet:
        show_route(route)

    print("\nassistant: ", end="", flush=True)
    empty = True
    pieces = []
    for piece in assistant.stream_answer(question, route, history=history, mode=mode):
        print(piece, end="", flush=True)
        pieces.append(piece)
        empty = False
    if empty:
        print("(no answer)", end="")
    print()

    answer = "".join(pieces).strip()
    if answer:
        history.append((question, answer))
    # Older turns are dropped by the token budget anyway; capping the list keeps the prompt
    # build cheap and stops one long session growing without limit.
    del history[:-MAX_HISTORY_TURNS]


def main():
    args = parse_args()
    device = pick_device(prefer_cpu=args.cpu)

    assistant = Assistant(device=device, quiet=args.quiet)
    print()

    try:
        if args.question:
            respond(assistant, " ".join(args.question), args.quiet, [], args.mode)
            return

        mode = args.mode
        print(f"Ask a question. Blank line or Ctrl-C to quit.  [mode: {mode}]")
        print("Commands: /chat  /ask  /auto  switch mode | /reset clears | /history shows context")
        history = []
        while True:
            try:
                question = input(f"\nuser ({mode}): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                break

            if question.lstrip("/") in config.MODES and question.startswith("/"):
                mode = question.lstrip("/")
                explain = {
                    "chat": "never retrieving - the whole context goes to the conversation",
                    "ask": "always retrieving - best for looking things up",
                    "auto": "the router decides per message",
                }[mode]
                print(f"(mode: {mode} - {explain})")
                continue
            if question == "/reset":
                history.clear()
                print("(conversation cleared)")
                continue
            if question == "/history":
                if not history:
                    print("(nothing yet - this is the start of the conversation)")
                for user_turn, assistant_turn in history:
                    print(f"  user      {user_turn}")
                    print(f"  assistant {assistant_turn}")
                continue
            respond(assistant, question, args.quiet, history, mode)
    finally:
        assistant.close()


if __name__ == "__main__":
    sys.exit(main())
