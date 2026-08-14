from datasets import load_dataset
from pathlib import Path

print("Downloading HellaSwag dataset (commonsense scenario completion)...")
dataset = load_dataset("Rowan/hellaswag")

output_dir = Path("data/hellaswag")
output_dir.mkdir(parents=True, exist_ok=True)

split_map = {"train": "train", "validation": "validation"}

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[hf_split]:
            label = example["label"]
            if label == "":
                continue
            ending = example["endings"][int(label)]
            f.write(f"User: What happens next? {example['ctx'].strip()}\nAssistant: {ending.strip()}\n<|endoftext|>\n")
    print(f"Done: {out_path}")
