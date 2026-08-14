import re
from datasets import load_dataset
from pathlib import Path

print("Downloading GSM8K dataset (grade-school math with step-by-step reasoning)...")
dataset = load_dataset("openai/gsm8k", "main")

output_dir = Path("data/gsm8k")
output_dir.mkdir(parents=True, exist_ok=True)

# GSM8K answers contain calculator annotations like <<48/2=24>> - strip them
# so the model learns natural step-by-step reasoning text, not notation.
calc_annotation = re.compile(r"<<[^>]*>>")

# GSM8K ships train/test splits; we treat test as our validation split.
split_map = {"train": "train", "test": "validation"}

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[hf_split]:
            question = example["question"].strip()
            answer = calc_annotation.sub("", example["answer"]).strip()
            f.write(f"User: {question}\nAssistant: {answer}\n<|endoftext|>\n")
    print(f"Done: {out_path}")
