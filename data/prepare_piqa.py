from datasets import load_dataset
from pathlib import Path

print("Downloading PIQA dataset (via parquet mirror, no remote code execution)...")
dataset = load_dataset("ybisk/piqa", revision="refs/convert/parquet")

output_dir = Path("data/piqa")
output_dir.mkdir(parents=True, exist_ok=True)

split_map = {"train": "train", "validation": "validation"}

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[hf_split]:
            goal = example["goal"].strip()
            solution = example["sol1"] if example["label"] == 0 else example["sol2"]
            # Both solutions look reasonable; without seeing them the model can only guess.
            options = "\n".join(f"- {s.strip()}" for s in (example["sol1"], example["sol2"]))
            f.write(f"User: {goal}\n\nOptions:\n{options}\nAssistant: {solution.strip()}\n<|endoftext|>\n")
    print(f"Done: {out_path}")
