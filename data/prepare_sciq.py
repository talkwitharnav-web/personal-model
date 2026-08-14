from datasets import load_dataset
from pathlib import Path

print("Downloading SciQ dataset (science Q&A with supporting fact passages)...")
dataset = load_dataset("allenai/sciq")

output_dir = Path("data/sciq")
output_dir.mkdir(parents=True, exist_ok=True)

split_map = {"train": "train", "validation": "validation"}

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[hf_split]:
            question = example["question"].strip()
            answer = example["correct_answer"].strip()
            support = example["support"].strip()

            if support:
                f.write(f"User: {question}\n\nContext: {support}\nAssistant: {answer}\n<|endoftext|>\n")
            else:
                f.write(f"User: {question}\nAssistant: {answer}\n<|endoftext|>\n")
    print(f"Done: {out_path}")
