from datasets import load_dataset
from pathlib import Path

print("Downloading CommonsenseQA dataset...")
dataset = load_dataset("tau/commonsense_qa")

output_dir = Path("data/commonsenseqa")
output_dir.mkdir(parents=True, exist_ok=True)

split_map = {"train": "train", "validation": "validation"}

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[hf_split]:
            answer_key = example["answerKey"]
            if not answer_key:
                continue
            idx = example["choices"]["label"].index(answer_key)
            answer_text = example["choices"]["text"][idx]
            # Without the choices the question is unanswerable - several options are
            # equally plausible, so the model would only learn to guess confidently.
            options = "\n".join(f"- {c.strip()}" for c in example["choices"]["text"])
            f.write(f"User: {example['question'].strip()}\n\nOptions:\n{options}\n"
                    f"Assistant: {answer_text.strip()}\n<|endoftext|>\n")
    print(f"Done: {out_path}")
