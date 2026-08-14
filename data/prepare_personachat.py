from datasets import load_dataset
from pathlib import Path
import re

print("Downloading Synthetic-Persona-Chat dataset (multi-turn persona-based conversation)...")
dataset = load_dataset("google/Synthetic-Persona-Chat")

output_dir = Path("data/personachat")
output_dir.mkdir(parents=True, exist_ok=True)

split_map = {"train": "train", "validation": "validation"}

# The conversation ships as one string with "User 1:"/"User 2:" turns already
# in it - relabel "User 1" -> "User" and "User 2" -> "Assistant".
turn_pattern = re.compile(r"^User (1|2):\s*(.*)$")

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[hf_split]:
            conversation = example["Best Generated Conversation"]
            lines = []
            for line in conversation.split("\n"):
                match = turn_pattern.match(line.strip())
                if not match:
                    continue
                speaker, text = match.groups()
                role = "User" if speaker == "1" else "Assistant"
                lines.append(f"{role}: {text.strip()}")
            if lines:
                f.write("\n".join(lines) + "\n<|endoftext|>\n")
    print(f"Done: {out_path}")
