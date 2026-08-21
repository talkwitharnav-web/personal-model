from datasets import load_dataset
from pathlib import Path

print("Downloading ReClor dataset (LSAT-style logical reasoning comprehension)...")
dataset = load_dataset("tasksource/reclor")

output_dir = Path("data/reclor")
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
            question = get_field(example, "question")
            options = get_field(example, "answers", "options")
            label = get_field(example, "label", "answer")

            if context is None or question is None or options is None or label is None or int(label) < 0:
                continue

            answer = options[int(label)]
            # Multiple choice - listing the options is what makes it answerable.
            option_lines = "\n".join(f"- {o.strip()}" for o in options)
            f.write(f"User: {context.strip()}\n{question.strip()}\n\nOptions:\n{option_lines}\n"
                    f"Assistant: {answer.strip()}\n<|endoftext|>\n")
    print(f"Done: {out_path}")
