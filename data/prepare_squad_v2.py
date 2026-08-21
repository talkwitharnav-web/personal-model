from datasets import load_dataset
from pathlib import Path

# SQuAD v2 adds ~43k questions whose answer is deliberately NOT in the passage.
# Those are the point of this dataset - they teach the model to decline instead
# of inventing something (like answering "eight" about the US flag's stars).
REFUSAL = "I don't know. The context does not say."

print("Downloading SQuAD v2 (includes unanswerable questions)...")
dataset = load_dataset("rajpurkar/squad_v2")

output_dir = Path("data/squad_v2")
output_dir.mkdir(parents=True, exist_ok=True)

for split in ["train", "validation"]:
    out_path = output_dir / f"{split}.txt"
    print(f"Writing {split} split to {out_path}...")
    answerable = 0
    unanswerable = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[split]:
            context = example["context"].strip()
            question = example["question"].strip()
            answers = example["answers"]["text"]

            if answers:
                answer = answers[0].strip()
                answerable += 1
            else:
                answer = REFUSAL
                unanswerable += 1

            f.write(f"User: {question}\n\nContext: {context}\nAssistant: {answer}\n<|endoftext|>\n")

    total = answerable + unanswerable
    print(f"Done: {out_path} - {total:,} examples "
          f"({answerable:,} answerable, {unanswerable:,} unanswerable = {unanswerable/total:.0%})")
