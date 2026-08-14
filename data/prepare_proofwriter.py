from datasets import load_dataset
from pathlib import Path

print("Downloading ProofWriter dataset (synthetic facts+rules deduction puzzles)...")
dataset = load_dataset("tasksource/proofwriter")

output_dir = Path("data/proofwriter")
output_dir.mkdir(parents=True, exist_ok=True)

# 585k train rows is far bigger than any other single fine-tune dataset here -
# subsample down to a proportionate size instead of ballooning the corpus.
SAMPLE_SIZES = {"train": 80000, "validation": 5000}

for hf_split, sample_size in SAMPLE_SIZES.items():
    out_path = output_dir / f"{hf_split}.txt"
    split_data = dataset[hf_split].shuffle(seed=42).select(range(sample_size))
    print(f"Writing {hf_split} split ({sample_size} of {len(dataset[hf_split])} rows) to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in split_data:
            theory = example["theory"].strip()
            question = example["question"].strip()
            answer = example["answer"].strip()
            f.write(f"User: {theory}\n{question}\nAssistant: {answer}\n<|endoftext|>\n")
    print(f"Done: {out_path}")
