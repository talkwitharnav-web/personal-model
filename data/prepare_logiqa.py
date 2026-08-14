from datasets import load_dataset
from pathlib import Path

print("Downloading LogiQA dataset (logical reasoning comprehension)...")
dataset = load_dataset("lucasmccabe/logiqa", revision="refs/convert/parquet")

output_dir = Path("data/logiqa")
output_dir.mkdir(parents=True, exist_ok=True)

def get_field(example, *candidate_keys):
    for key in candidate_keys:
        if key in example and example[key] is not None:
            return example[key]
    return None

split_map = {"train": "train", "validation": "validation"}

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[hf_split]:
            context = get_field(example, "context")
            query = get_field(example, "query")
            options = get_field(example, "options", "answers")
            correct_index = get_field(example, "correct_option", "label")

            if context is None or query is None or options is None or correct_index is None:
                continue

            answer = options[int(correct_index)]
            f.write(f"User: {context.strip()}\n{query.strip()}\nAssistant: {answer.strip()}\n<|endoftext|>\n")
    print(f"Done: {out_path}")
