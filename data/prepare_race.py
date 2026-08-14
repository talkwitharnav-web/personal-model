from datasets import load_dataset
from pathlib import Path

print("Downloading RACE dataset (exam reading comprehension)...")
dataset = load_dataset("ehovy/race", "all")

output_dir = Path("data/race")
output_dir.mkdir(parents=True, exist_ok=True)

split_map = {"train": "train", "validation": "validation"}

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[hf_split]:
            answer_letter = example["answer"]
            if answer_letter not in "ABCD":
                continue
            idx = "ABCD".index(answer_letter)
            answer = example["options"][idx].strip()
            article = example["article"].strip()
            question = example["question"].strip()
            f.write(f"User: {article}\n\n{question}\nAssistant: {answer}\n<|endoftext|>\n")
    print(f"Done: {out_path}")
