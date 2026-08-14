from datasets import load_dataset
from pathlib import Path

print("Downloading Dolly-15k dataset (general instruction-following)...")
dataset = load_dataset("databricks/databricks-dolly-15k")

output_dir = Path("data/dolly")
output_dir.mkdir(parents=True, exist_ok=True)

# Dolly only ships one "train" split - hold out the last 5% ourselves as validation
all_rows = list(dataset["train"])
split_index = int(len(all_rows) * 0.95)
split_map = {"train": all_rows[:split_index], "validation": all_rows[split_index:]}

for out_split, rows in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {out_split} split ({len(rows)} rows) to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in rows:
            instruction = example["instruction"].strip()
            context = example["context"].strip()
            response = example["response"].strip()

            if context:
                f.write(f"User: {instruction}\n\nContext: {context}\nAssistant: {response}\n<|endoftext|>\n")
            else:
                f.write(f"User: {instruction}\nAssistant: {response}\n<|endoftext|>\n")
    print(f"Done: {out_path}")
