from datasets import load_dataset
from pathlib import Path

print("Downloading Social IQa dataset (via parquet mirror, no remote code execution)...")
dataset = load_dataset("allenai/social_i_qa", revision="refs/convert/parquet")

output_dir = Path("data/socialiqa")
output_dir.mkdir(parents=True, exist_ok=True)

split_map = {"train": "train", "validation": "validation"}

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[hf_split]:
            answer_map = {"1": example["answerA"], "2": example["answerB"], "3": example["answerC"]}
            answer = answer_map.get(example["label"].strip())
            if answer is None:
                continue
            # Without the choices the question is unanswerable - several options are
            # equally plausible, so the model would only learn to guess confidently.
            options = "\n".join(f"- {answer_map[k].strip()}" for k in ("1", "2", "3"))
            f.write(f"User: {example['context'].strip()} {example['question'].strip()}\n\n"
                    f"Options:\n{options}\nAssistant: {answer.strip()}\n<|endoftext|>\n")
    print(f"Done: {out_path}")
