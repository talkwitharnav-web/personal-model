from datasets import load_dataset
from pathlib import Path

print("Downloading XSum dataset (news article summarization)...")
dataset = load_dataset("EdinburghNLP/xsum")

output_dir = Path("data/xsum")
output_dir.mkdir(parents=True, exist_ok=True)

split_map = {"train": "train", "validation": "validation"}

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[hf_split]:
            document = example["document"].strip()
            summary = example["summary"].strip()
            f.write(f"User: Summarize this: {document}\nAssistant: {summary}\n<|endoftext|>\n")
    print(f"Done: {out_path}")
