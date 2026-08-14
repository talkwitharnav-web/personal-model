from datasets import load_dataset
from pathlib import Path

print("Downloading SQuAD dataset (extractive reading comprehension)...")
dataset = load_dataset("rajpurkar/squad")

output_dir = Path("data/squad")
output_dir.mkdir(parents=True, exist_ok=True)

split_map = {"train": "train", "validation": "validation"}

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[hf_split]:
            answers = example["answers"]["text"]
            if not answers:
                continue
            answer = answers[0].strip()
            context = example["context"].strip()
            question = example["question"].strip()
            f.write(f"User: {question}\n\nContext: {context}\nAssistant: {answer}\n<|endoftext|>\n")
    print(f"Done: {out_path}")
