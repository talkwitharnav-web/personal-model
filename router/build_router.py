from pathlib import Path
import random

CATEGORIES = {
    "maths": ["gsm8k"],
    "summarise": ["xsum"],
    "factual": ["squad", "boolq", "sciq", "race"],
    "commonsense": ["piqa", "commonsenseqa", "socialiqa", "hellaswag"],
    "reasoning": ["logiqa", "reclor", "proofwriter", "strategyqa"],
    "chitchat": ["personachat", "dailydialog", "oasst1"],
    "instruction": ["dolly"],
}

SAMPLES_PER_CATEGORY = 3000

# xsum is the only set that puts its instruction BEFORE the material,
# so its question is the first line, not the last.
INSTRUCTION_FIRST = {"xsum"}

# Conversations, so any user turn is a fair sample - always taking the last one
# would teach the router that chitchat means "goodbye".
MULTI_TURN = {"personachat", "dailydialog", "oasst1"}

# Every xsum example opens with this exact string, which the classifier would
# memorise instead of learning what a summary request looks like.
SUMMARISE_MARKER = "Summarize this:"
SUMMARISE_PROMPTS = [
    "Summarize this:",
    "Summarise this article:",
    "Give me a summary of this:",
    "TL;DR:",
    "Can you sum this up?",
    "What's the gist of this?",
    "Briefly, what does this say?",
    "Shorten this for me:",
    "Give me the key points of this:",
    "In a sentence, what is this about?",
]

_rng = random.Random(1234)

def iter_examples(path, read_bytes=10_000_000):
    buffer = ""
    with open(path, "r", encoding="utf-8") as f:
        while True:
            data = f.read(read_bytes)
            if not data:
                break
            buffer += data
            parts = buffer.split("<|endoftext|>")
            buffer = parts.pop()
            for example in parts:
                yield example
    if buffer.strip():
        yield buffer

def reservoir_sample(items, sample_size, seed=42):
    rng = random.Random(seed)
    reservoir = []
    for count, item in enumerate(items):
        if count < sample_size:
            reservoir.append(item)
        else:
            slot = rng.randint(0, count)
            if slot < sample_size:
                reservoir[slot] = item
    return reservoir

def user_messages(user_turn):
    messages = []
    for position, line in enumerate(user_turn.split("\n")):
        line = line.strip()
        if line.startswith("User:"):
            line = line[len("User:"):].strip()
        elif position != 0:
            continue  # assistant text, or a continuation of it
        if line:
            messages.append(line)
    return messages

def extract_question(example, dataset_name):
    start = example.find("User:")
    end = example.rfind("\nAssistant:")
    if start == -1 or end == -1:
        return None

    user_turn = example[start + len("User:"):end].strip()

    if dataset_name in MULTI_TURN:
        messages = user_messages(user_turn)
        if not messages:
            return None
        question = _rng.choice(messages)
    elif "\n\nContext:" in user_turn:
        question = user_turn.split("\n\nContext:")[0]
    elif dataset_name in INSTRUCTION_FIRST:
        question = user_turn.split("\n")[0]
    else:
        question = user_turn.split("\n")[-1]

    question = question.strip()
    if question.startswith("User:"):
        question = question[len("User:"):].strip()

    if question.startswith(SUMMARISE_MARKER):
        body = question[len(SUMMARISE_MARKER):].strip()
        question = f"{_rng.choice(SUMMARISE_PROMPTS)} {body}"

    return question if len(question) > 5 else None

def iter_questions(dataset_name):
    path = Path("data") / dataset_name / "train.txt"
    for example in iter_examples(path):
        question = extract_question(example, dataset_name)
        if question is not None:
            yield question


def build_dataset():
    questions = []
    labels = []
    for category, folders in CATEGORIES.items():
        per_folder = SAMPLES_PER_CATEGORY // len(folders)
        for folder in folders:
            sample = reservoir_sample(iter_questions(folder), per_folder)
            questions.extend(sample)
            labels.extend([category] * len(sample))
            print(f"{category:12} <- {folder:14} {len(sample):5,} questions")
    return questions, labels

if __name__ == "__main__":
    questions, labels = build_dataset()
    print(f"\nTotal: {len(questions):,} questions, {len(set(labels))} categories")

    print("\n--- one sample per category ---")
    for category in CATEGORIES:
        i = labels.index(category)
        print(f"[{category}] {questions[i][:120]}")