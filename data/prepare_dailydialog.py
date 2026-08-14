import re
from datasets import load_dataset
from pathlib import Path

print("Downloading DailyDialog dataset (via parquet mirror, no remote code execution)...")
dataset = load_dataset("li2017dailydialog/daily_dialog", revision="refs/convert/parquet")

output_dir = Path("data/dailydialog")
output_dir.mkdir(parents=True, exist_ok=True)

# Small curated profanity list - words are removed entirely, not asterisk-censored
PROFANITY = [
    "fuck", "shit", "bitch", "asshole", "bastard", "damn", "crap", "dick",
    "piss", "cunt", "whore", "slut", "douche", "prick", "twat",
]
profanity_pattern = re.compile(r"\b(" + "|".join(PROFANITY) + r")\w*\b", re.IGNORECASE)

def clean_text(text):
    cleaned = profanity_pattern.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()

split_map = {"train": "train", "validation": "validation"}

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[hf_split]:
            lines = []
            for i, turn in enumerate(example["dialog"]):
                speaker = "User" if i % 2 == 0 else "Assistant"
                text = clean_text(turn)
                if text:
                    lines.append(f"{speaker}: {text}")
            if lines:
                f.write("\n".join(lines) + "\n<|endoftext|>\n")
    print(f"Done: {out_path}")
