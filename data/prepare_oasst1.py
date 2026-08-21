from datasets import load_dataset
from pathlib import Path
import re

print("Downloading OASST1 dataset (real multi-turn, topic-varied assistant conversations)...")
dataset = load_dataset("OpenAssistant/oasst1")

output_dir = Path("data/oasst1")
output_dir.mkdir(parents=True, exist_ok=True)

split_map = {"train": "train", "validation": "validation"}

# At most this many alternative replies to the same conversation prefix.
MAX_REPLIES_PER_PREFIX = 2

# OASST1 is the Open Assistant project's own dataset, so its assistant talks about itself by that
# name - 308 assistant lines mention it, 104 of them self-identifications ("I'm Open Assistant").
# That is a COMPETING IDENTITY. Measured 2026-08-21 on the run-4 model: even with DarkGLM
# outnumbering "Open Assistant" 2,896 to 140 in the mix (21:1), "what is your name?" still came
# back "My name is Open Assistant." on 3 of 5 samples. Out-voting a competing identity does not
# work; removing it does.
#
# Renaming rather than deleting keeps the content - "DarkGLM is an open source alternative to
# ChatGPT" is a perfectly good thing for this model to say about itself, and it now REINFORCES
# the identity instead of contradicting it.
ASSISTANT_NAME = "DarkGLM"
IDENTITY_PATTERN = re.compile(r"Open\s?Assistant", re.IGNORECASE)

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")

    messages_by_id = {}
    for example in dataset[hf_split]:
        if example["lang"] == "en" and not example["deleted"]:
            messages_by_id[example["message_id"]] = example

    with open(out_path, "w", encoding="utf-8") as f:
        kept = skipped_rank = skipped_cap = 0
        prefix_seen = {}
        for msg in messages_by_id.values():
            if msg["role"] != "assistant":
                continue

            # OASST1 ranks the sibling replies to each prompt by human preference: rank 0 is the
            # one annotators chose, higher numbers are the ones they liked less. The first build
            # took ALL of them, so 23,427 examples covered only 8,923 distinct conversation
            # prefixes - 47.9% outright duplicates, one prefix used 14 times - and most of those
            # replies had been explicitly judged worse by the people who read them. Training on a
            # reply that lost its own comparison teaches the losing behaviour, which is the
            # opposite of what an instruction set is for.
            #
            # Keeping the top two rather than only rank 0 is a deliberate middle: rank 0 alone
            # would cut this set to about 3,500 and instruction following is the capability it
            # carries. Replies with no rank are singletons with nothing to compare against, so
            # they are kept - but the per-prefix cap below still applies to them, because a great
            # many carry no rank at all and would otherwise rebuild the same tail.
            rank = msg.get("rank")
            if rank is not None and rank > 1:
                skipped_rank += 1
                continue

            # Walk back up the parent chain to reconstruct the full conversation so far.
            chain = []
            current = msg
            valid = True
            while current is not None:
                chain.append(current)
                parent_id = current["parent_id"]
                if parent_id is None:
                    break
                current = messages_by_id.get(parent_id)
                if current is None:
                    valid = False
                    break

            if not valid:
                continue

            chain.reverse()
            lines = []
            for turn in chain:
                role = "User" if turn["role"] == "prompter" else "Assistant"
                text = IDENTITY_PATTERN.sub(ASSISTANT_NAME, turn["text"].strip())
                lines.append(f"{role}: {text}")

            # Two alternative answers to the same prompt is useful variety; fourteen is one
            # prompt drowning out thirteen others.
            prefix = "\n".join(lines[:-1])
            if prefix_seen.get(prefix, 0) >= MAX_REPLIES_PER_PREFIX:
                skipped_cap += 1
                continue
            prefix_seen[prefix] = prefix_seen.get(prefix, 0) + 1

            f.write("\n".join(lines) + "\n<|endoftext|>\n")
            kept += 1
    print(f"Done: {out_path} - {kept:,} conversations kept, "
          f"{skipped_rank:,} dropped for being ranked below the top two, "
          f"{skipped_cap:,} dropped by the {MAX_REPLIES_PER_PREFIX}-per-prompt cap")
