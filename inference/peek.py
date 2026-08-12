# Quick, throwaway script to peek at the model's current output mid-training.
# Not the "real" inference lesson (that comes later) - just a sanity check.
import torch
from tokenizers import Tokenizer
from model.config import DarkGLMConfig
from model.model import DarkGLM

device = "cuda" if torch.cuda.is_available() else "cpu"
config = DarkGLMConfig()

model = DarkGLM(config).to(device)
checkpoint = torch.load("checkpoints/latest.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()
print(f"Loaded checkpoint from step {checkpoint['step']}")

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

prompt = "Once upon a time, there was a"
input_ids = tokenizer.encode(prompt).ids
input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

max_new_tokens = 700
temperature = 0.8
top_k = 50

with torch.no_grad():
    for _ in range(max_new_tokens):
        input_crop = input_ids[:, -config.context_length:]
        logits = model(input_crop)
        logits = logits[:, -1, :] / temperature

        top_values, top_indices = torch.topk(logits, top_k)
        probs = torch.softmax(top_values, dim=-1)
        next_token = top_indices.gather(-1, torch.multinomial(probs, num_samples=1))

        input_ids = torch.cat([input_ids, next_token], dim=1)

output_text = tokenizer.decode(input_ids[0].tolist())
print("\n--- Generated text ---")
print(output_text)
