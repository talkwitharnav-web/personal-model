"""Measure the retrieval fix properly: many samples, both decoders, before vs after.

A single sample cannot tell "got worse" from "rolled badly" - `tools/peek_variance.py` established
that on 2026-08-18 and it has been re-learned twice since. Eyeballing one run of the pipeline after
the retrieval change produced exactly that trap: "capital of France" clearly improved
("Founded in the 19th century" -> "Paris") while "what colour is the sky" appeared to regress
("azure" -> "orange"), which is a temperature artifact, not a change in the retriever.

So this asks each question N times and reports a hit RATE, plus one greedy answer as the
deterministic reference point.

Run it with `--tag before` on the old retriever and `--tag after` on the new one to compare, or
just read the rates - anything below about 0.8 is unreliable regardless of what one sample showed.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from orchestrator.pipeline import Assistant

# (question, accepted answer fragments). Several fragments where more than one wording is right.
CASES = [
    ("What is the capital of France?", ["paris"]),
    ("Who invented the Tesla coil?", ["nikola tesla", "tesla"]),
    ("When did the Second World War begin?", ["1939"]),
    ("What is photosynthesis?", ["light energy", "chemical energy", "sunlight", "carbon dioxide"]),
    ("What is a mitochondrion?", ["organelle"]),
    ("How many legs does a cat have?", ["four", "4"]),
    ("What colour is the sky on a clear day?", ["blue", "azure"]),
    ("What do bees make?", ["honey"]),
    ("Was penicillin discovered by Alexander Fleming?", ["yes", "fleming"]),
    ("What is an earthquake?", ["shaking", "seismic", "energy"]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="after")
    args = parser.parse_args()

    assistant = Assistant(device=args.device, quiet=True)
    print(f"[{args.tag}] {args.samples} samples per question, plus one greedy answer\n")

    total_rate = 0.0
    greedy_hits = 0
    dropped_total = 0
    for question, accepted in CASES:
        route = assistant.plan(question)
        dropped_total += route.dropped
        prompt, _ = assistant.generator.build_prompt(question, route.passages)

        hits = 0
        answers = Counter()
        for _ in range(args.samples):
            answer = "".join(assistant.generator.stream(prompt)).strip()
            answers[answer[:60]] += 1
            if any(fragment in answer.lower() for fragment in accepted):
                hits += 1
        # Greedy: temperature near zero is the deterministic reference.
        greedy = "".join(assistant.generator.stream(prompt, temperature=0.01)).strip()
        greedy_hit = any(fragment in greedy.lower() for fragment in accepted)
        greedy_hits += greedy_hit

        rate = hits / args.samples
        total_rate += rate
        print(f"  {rate:4.0%} sampled | {'HIT ' if greedy_hit else 'miss'} greedy | "
              f"dropped {route.dropped:2d} | {question}")
        print(f"         greedy: {greedy[:88]}")
        if rate not in (0.0, 1.0):
            common = ", ".join(f"{text!r} x{count}" for text, count in answers.most_common(2))
            print(f"         varies: {common}")

    n = len(CASES)
    print(f"\n{'=' * 92}")
    print(f"  [{args.tag}] mean sampled hit rate {total_rate/n:.1%} | greedy {greedy_hits}/{n}")
    print(f"  boilerplate/duplicate passages dropped across all queries: {dropped_total}")
    print("=" * 92)
    assistant.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
