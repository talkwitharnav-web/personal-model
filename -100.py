from tokenizers import Tokenizer
from pathlib import Path
from model.config import DarkGLMConfig

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
example = "user: What is 2+2?\nAssistant: 4\n"
encoding = tokenizer.encode(example)

assistant_marker_ids = tokenizer.encode("\nAssistant:").ids

marker = assistant_marker_ids[1:]
first_example = text.split("<|endoftext|>")[0]

config = DarkGLMConfig()
pad_id = tokenizer.token_to_id("<|endoftext|>")

def tokenize_example(text):
    full_ids = tokenizer.encode(text).ids

    for i in range(len(full_ids) - len(marker) + 1):
        if full_ids[i:i + len(marker)] == marker:
            answer_start = i + len(marker)
            break
    else:
        return None

    labels = list(full_ids)
    for i in range(answer_start):
        labels[i] = -100

    if len(full_ids) > config.context_length:
        return None

    pad_amount = config.context_length - len(full_ids)
    full_ids = full_ids + [pad_id] * pad_amount
    labels = labels + [-100] * pad_amount

    return full_ids, labels

result = tokenize_example(first_example)
print(len(result[0]), len(result[1]))
print(result[00][:20])
print(result[1][:20])