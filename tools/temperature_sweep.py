# Runs a fixed question set at several temperatures so settings can be chosen from
# evidence rather than guesswork.
import sys

from orchestrator.generation import pick_device
from orchestrator.pipeline import Assistant

TEMPERATURES = [0.1, 0.3, 0.6, 0.9]

QUESTIONS = [
    "who is william shakespeare?",
    "what is a cow",
    "what is a tesla coil",
    "Who created Minecraft?",
    "What is photosynthesis?",
]


def main():
    assistant = Assistant(device=pick_device(), quiet=True)
    print()
    try:
        for question in QUESTIONS:
            route = assistant.plan(question)
            print(f"=== {question}   [{route.category} {route.confidence:.2f}]")
            if route.passages:
                print(f"    top passage: {route.passages[0][1][:110]}")
            for temperature in TEMPERATURES:
                answer = assistant.answer(question, route, temperature=temperature)
                print(f"    t={temperature}  {answer[:150]}")
            print()
    finally:
        assistant.close()


if __name__ == "__main__":
    sys.exit(main())
