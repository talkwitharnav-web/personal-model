from datasets import load_dataset
from pathlib import Path
import random
import re

print("Downloading Synthetic-Persona-Chat dataset (multi-turn persona-based conversation)...")
dataset = load_dataset("google/Synthetic-Persona-Chat")

output_dir = Path("data/personachat")
output_dir.mkdir(parents=True, exist_ok=True)

split_map = {"train": "train", "validation": "validation"}

# The conversation ships as one string with "User 1:"/"User 2:" turns already
# in it - relabel "User 1" -> "User" and "User 2" -> "Assistant".
turn_pattern = re.compile(r"^User (1|2):\s*(.*)$")

# 32.7% of these synthetic conversations ship with the generator's placeholders still in them -
# "Hi! I'm [user 1's name]." - and training on that teaches the model to emit literal [name] in
# chat. Measured counts: [name] 1,503, [user 2's name] 1,013, [user 1's name] 967, and a long tail
# of casing and spacing variants. They are substituted with real names, consistently within each
# conversation so the two speakers stay distinguishable; anything left unresolved drops the row.
NAMES = [
    "Alex", "Sam", "Jordan", "Casey", "Riley", "Morgan", "Taylor", "Jamie", "Avery", "Quinn",
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason", "Isabella", "Lucas",
    "Mia", "James", "Charlotte", "Ben", "Amelia", "Henry", "Harper", "Jack", "Ella", "Owen",
    "Grace", "Leo", "Chloe", "Nathan", "Zoe", "Adam", "Lily", "Daniel", "Nora", "Caleb",
]

PLACEHOLDER = re.compile(r"\[[^\]]{1,40}\]")


def fill_placeholders(text, speaker_name, other_name):
    """Replace [user 1's name] / [user 2's name] / [name] with real names.

    Markers are used for the numbered forms rather than the names directly, because which
    speaker is "user 1" is fixed for the whole conversation while `speaker_name` changes line
    by line. Substituting the markers once at the end keeps both speakers consistent.
    """
    text = re.sub(r"\[\s*user\s*1'?s?\s*name\s*\]", FIRST_MARKER, text, flags=re.I)
    text = re.sub(r"\[\s*user\s*2'?s?\s*name\s*\]", SECOND_MARKER, text, flags=re.I)
    text = re.sub(r"\[\s*user\s*1\s*\]", FIRST_MARKER, text, flags=re.I)
    text = re.sub(r"\[\s*user\s*2\s*\]", SECOND_MARKER, text, flags=re.I)
    # A bare [name] is the SPEAKER naming themselves ("Hi! I'm [name]").
    text = re.sub(r"\[\s*name\s*\]", speaker_name, text, flags=re.I)
    text = re.sub(r"\[\s*your\s+name\s*\]", other_name, text, flags=re.I)
    return text


FIRST_MARKER = "\x00FIRST\x00"
SECOND_MARKER = "\x00SECOND\x00"

for hf_split, out_split in split_map.items():
    out_path = output_dir / f"{out_split}.txt"
    print(f"Writing {hf_split} split to {out_path}...")
    rng = random.Random(11)
    written = dropped = substituted = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for example in dataset[hf_split]:
            conversation = example["Best Generated Conversation"]
            name_one, name_two = rng.sample(NAMES, 2)
            lines = []
            for line in conversation.split("\n"):
                match = turn_pattern.match(line.strip())
                if not match:
                    continue
                speaker, text = match.groups()
                role = "User" if speaker == "1" else "Assistant"
                own = name_one if speaker == "1" else name_two
                other = name_two if speaker == "1" else name_one
                if "[" in text:
                    text = fill_placeholders(text, own, other)
                lines.append(f"{role}: {text.strip()}")

            body = "\n".join(lines)
            if FIRST_MARKER in body or SECOND_MARKER in body:
                substituted += 1
            body = body.replace(FIRST_MARKER, name_one).replace(SECOND_MARKER, name_two)
            # Anything still bracketed was a placeholder shape not covered above. Dropping is
            # safer than shipping it: a model that learns to type "[" mid-sentence is worse than
            # a slightly smaller dataset.
            if PLACEHOLDER.search(body):
                dropped += 1
                continue
            lines = body.split("\n")

            # Training masks everything up to the LAST "Assistant:", so a conversation
            # ending on a User turn would teach the model to write the user's next line
            # too. Cut back to the final Assistant reply.
            while lines and lines[-1].startswith("User:"):
                lines.pop()

            if len(lines) >= 2:
                f.write("\n".join(lines) + "\n<|endoftext|>\n")
                written += 1
    print(f"Done: {out_path} - {written:,} conversations "
          f"({substituted:,} had placeholders filled, {dropped:,} dropped as unresolvable)")
