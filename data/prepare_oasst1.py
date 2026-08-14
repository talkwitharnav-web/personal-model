from datasets import load_dataset
from pathlib import Path

print("Downloading OASST1 dataset (real multi-turn, topic-varied assistant conversations)...")
dataset = load_dataset("OpenAssistant/oasst1")

output_dir = Path("data/oasst1")
output_dir.mkdir(parents=True, exist_ok=True)

split_map = {"train": "train", "validation": "validation"}

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")

    messages_by_id = {}
    for example in dataset[hf_split]:
        if example["lang"] == "en" and not example["deleted"]:
            messages_by_id[example["message_id"]] = example

    with open(out_path, "w", encoding="utf-8") as f:
        for msg in messages_by_id.values():
            if msg["role"] != "assistant":
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
                lines.append(f"{role}: {turn['text'].strip()}")
            f.write("\n".join(lines) + "\n<|endoftext|>\n")
    print(f"Done: {out_path}")
