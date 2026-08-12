from datasets import load_dataset
from pathlib import Path
print("Downloading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories")
output_dir = Path("data/tinystories")
output_dir.mkdir(parents=True, exist_ok=True)
for split_name in ["train", "validation"]:
    out_path = output_dir / f"{split_name}.txt"
    print(f"Writing {split_name} split to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[split_name]:
            f.write(example["text"].strip() + "\n<|endoftext|>\n")
    print(f"Done: {out_path}")