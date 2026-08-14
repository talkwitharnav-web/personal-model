from datasets import load_dataset
from pathlib import Path

print("Downloading WikiText-103 dataset...")
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

output_dir = Path("data/wikitext")
output_dir.mkdir(parents=True, exist_ok=True)

for split_name in ["train", "validation"]:
    out_path = output_dir / f"{split_name}.txt"
    print(f"Writing {split_name} split to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[split_name]:
            text = example["text"].strip()
            if text:
                f.write(text + "\n<|endoftext|>\n")
    print(f"Done: {out_path}")
