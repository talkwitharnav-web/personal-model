from datasets import load_dataset
from pathlib import Path
import random

print("Downloading BoolQ dataset (naturally-occurring yes/no questions with evidence)...")
dataset = load_dataset("google/boolq")

output_dir = Path("data/boolq")
output_dir.mkdir(parents=True, exist_ok=True)

split_map = {"train": "train", "validation": "validation"}

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")

    # BoolQ is 62% "yes". Trained on that, a model too weak to work the answer out
    # just guesses "yes" every time - which is exactly what happened. Downsample the
    # majority so there is no prior worth exploiting.
    yes = [e for e in dataset[hf_split] if e["answer"]]
    no = [e for e in dataset[hf_split] if not e["answer"]]
    keep = min(len(yes), len(no))
    rng = random.Random(42)
    balanced = rng.sample(yes, keep) + rng.sample(no, keep)
    rng.shuffle(balanced)

    with open(out_path, "w", encoding="utf-8") as f:
        for example in balanced:
            question = example["question"].strip()
            passage = example["passage"].strip()
            answer = "Yes" if example["answer"] else "No"
            f.write(f"User: {question}\n\nContext: {passage}\nAssistant: {answer}\n<|endoftext|>\n")
    print(f"Done: {out_path} - {len(balanced):,} examples, {keep:,} yes / {keep:,} no")
    print(f"Done: {out_path}")
