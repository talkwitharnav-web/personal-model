# Quick, throwaway script to peek at the model's current output mid-training.
# Not the "real" inference lesson (that comes later) - just a sanity check.
import torch
from tokenizers import Tokenizer
from model.config import DarkGLMConfig
from model.model import DarkGLM

# CPU on purpose: combined training is using nearly all 8GB of VRAM right now.
device = "cpu"
config = DarkGLMConfig()

model = DarkGLM(config).to(device)
model.load_state_dict(torch.load("checkpoints/combined_best_model.pt", map_location=device))
model.eval()
print("Loaded checkpoints/combined_best_model.pt")

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

# Instruction format the combined dataset was trained on
prompt = "User: What's the temperature in the arctic?\nAssistant:"
input_ids = tokenizer.encode(prompt).ids
input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

max_new_tokens = 100
temperature = 0.8
top_k = 50
repetition_penalty = 1.3
endoftext_id = tokenizer.token_to_id("<|endoftext|>")

with torch.no_grad():
    for _ in range(max_new_tokens):
        input_crop = input_ids[:, -config.context_length:]
        logits = model(input_crop)
        logits = logits[:, -1, :] / temperature

        # push down scores of tokens already used, so it can't loop on the same phrase forever
        for token_id in set(input_ids[0].tolist()):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

        top_values, top_indices = torch.topk(logits, top_k)
        probs = torch.softmax(top_values, dim=-1)
        next_token = top_indices.gather(-1, torch.multinomial(probs, num_samples=1))

        input_ids = torch.cat([input_ids, next_token], dim=1)

        # stop as soon as the model itself signals "done", same token that ends every training example
        if next_token.item() == endoftext_id:
            break

output_text = tokenizer.decode(input_ids[0].tolist())
print("\n--- Generated text ---")
print(output_text)
