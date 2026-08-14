from datasets import load_dataset
from pathlib import Path

print("Downloading StrategyQA dataset (implicit multi-step reasoning)...")
dataset = load_dataset("ChilleD/StrategyQA")

output_dir = Path("data/strategyqa")
output_dir.mkdir(parents=True, exist_ok=True)

def get_field(example, *candidate_keys):
    for key in candidate_keys:
        if key in example and example[key]:
            return example[key]
    return None

# StrategyQA only ships train/test splits - treat test as our validation split
split_map = {"train": "train", "test": "validation"}

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[hf_split]:
            question = get_field(example, "question")
            answer = example.get("answer")  # boolean False is a valid answer, not "missing"
            reasoning_steps = get_field(example, "facts", "decomposition", "evidence")

            if question is None or answer is None:
                continue

            answer_text = "Yes" if answer in (True, "True", "true", 1) else "No"

            if reasoning_steps:
                if isinstance(reasoning_steps, str):
                    reasoning = reasoning_steps.strip()
                else:
                    reasoning = " ".join(str(step).strip() for step in reasoning_steps if str(step).strip())
                f.write(f"User: {question.strip()}\nAssistant: {reasoning} Answer: {answer_text}\n<|endoftext|>\n")
            else:
                f.write(f"User: {question.strip()}\nAssistant: {answer_text}\n<|endoftext|>\n")
    print(f"Done: {out_path}")
