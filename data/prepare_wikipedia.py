from datasets import load_dataset
from pathlib import Path

# wikimedia/wikipedia has no per-article timestamp field (id, url, title, text only),
# so we random-sample via a shuffle buffer over the streamed dataset instead.
print("Streaming wikimedia/wikipedia (20231101.en)...")
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
dataset = dataset.shuffle(seed=42, buffer_size=10000)

output_dir = Path("data/wikipedia")
output_dir.mkdir(parents=True, exist_ok=True)

# Rough English BPE average is ~4.2 chars/token; budget with overshoot so the
# tokenize step (which knows exact token counts) always has enough raw text to
# hit its real targets (1.1B train tokens, 3M validation tokens) with room to spare.
CHARS_PER_TOKEN_ESTIMATE = 4.5
OVERSHOOT = 1.15
TRAIN_TOKEN_TARGET = 1_100_000_000
VALID_TOKEN_TARGET = 3_000_000
TRAIN_CHAR_BUDGET = int(TRAIN_TOKEN_TARGET * CHARS_PER_TOKEN_ESTIMATE * OVERSHOOT)
VALID_CHAR_BUDGET = int(VALID_TOKEN_TARGET * CHARS_PER_TOKEN_ESTIMATE * OVERSHOOT)

print(f"Train char budget: {TRAIN_CHAR_BUDGET:,}  Validation char budget: {VALID_CHAR_BUDGET:,}")

train_path = output_dir / "train.txt"
valid_path = output_dir / "validation.txt"

chars_written = 0
articles_written = 0
train_f = open(train_path, "w", encoding="utf-8")
target_f = train_f
target_budget = TRAIN_CHAR_BUDGET
switched_to_valid = False

print(f"Writing {train_path}...")
for example in dataset:
    text = example["text"].strip()
    if not text:
        continue
    target_f.write(text + "\n<|endoftext|>\n")
    chars_written += len(text)
    articles_written += 1

    if articles_written % 5000 == 0:
        print(f"  {articles_written:,} articles, {chars_written:,} chars written (current target: {target_f.name})")

    if chars_written >= target_budget:
        if not switched_to_valid:
            train_f.close()
            print(f"Train budget reached: {articles_written:,} articles, {chars_written:,} chars -> {train_path}")
            target_f = open(valid_path, "w", encoding="utf-8")
            print(f"Writing {valid_path}...")
            switched_to_valid = True
            chars_written = 0
            target_budget = VALID_CHAR_BUDGET
        else:
            target_f.close()
            print(f"Validation budget reached: {chars_written:,} chars -> {valid_path}")
            break

print("Done.")
