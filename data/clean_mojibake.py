import os

REPLACEMENTS = [
    ("â€™", "\u2019"),  # '
    ("â€˜", "\u2018"),  # '
    ("â€œ", "\u201c"),  # "
    ("â€”", "\u2014"),  # em dash
    ("â€“", "\u2013"),  # en dash
    ("â€¦", "\u2026"),  # ellipsis
    ("â€", '"'),        # leftover closing double-quote byte was dropped in the source data; best-effort plain quote
]

def fix_mojibake(text):
    for bad, good in REPLACEMENTS:
        text = text.replace(bad, good)
    return text

def clean_file(path):
    print(f"Cleaning {path}...")
    with open(path, encoding="utf-8") as f:
        text = f.read()

    fixed = fix_mojibake(text)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(fixed)

    os.replace(tmp_path, path)
    print(f"Done: {path}")

clean_file("data/tinystories/train.txt")
clean_file("data/tinystories/validation.txt")
