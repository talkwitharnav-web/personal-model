import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.config import DarkGLMConfig
from model.model import DarkGLM

config = DarkGLMConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = DarkGLM(config).to(device)
model.gradient_checkpointing = True
train_data = np.memmap("data/tinystories/train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("data/tinystories/validation.bin", dtype=np.uint16, mode="r")

def get_batch(split, batch_size):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - config.context_length, (batch_size,))

    x = torch.stack([torch.from_numpy(data[i:i + config.context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + config.context_length].astype(np.int64)) for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y

eval_every = 250
eval_iters = 20

best_val_loss = float("inf")
patience = 5
patience_counter = 0

@torch.no_grad()
def estimate_val_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch("validation", batch_size)
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = torch.amp.GradScaler(device)

batch_size = 16

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
save_every = 500

start_step = 0
checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    start_step = checkpoint["step"] + 1
    best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
    patience_counter = checkpoint.get("patience_counter", patience_counter)
    print(f"Resumed from checkpoint at step {start_step} (best_val_loss={best_val_loss:.4f}, patience={patience_counter}/{patience})")

grad_accum_steps = 4
max_steps = 38316  # one full epoch over train.bin at batch_size=16, grad_accum=4
log_every = 100

model.train()
optimizer.zero_grad()

for step in range(start_step, max_steps):
    x, y = get_batch("train", batch_size)

    with torch.autocast(device_type=device, dtype=torch.float16):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        loss = loss / grad_accum_steps

    scaler.scale(loss).backward()

    if (step + 1) % grad_accum_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    if step % log_every == 0:
        print(f"Step {step}/{max_steps} | Loss: {loss.item() * grad_accum_steps:.4f}")

    if step % eval_every == 0:
        val_loss = estimate_val_loss()
        print(f"Step {step}/{max_steps} | Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Validation loss stopped improving. Stopping training early.")
            break

    if (step + 1) % save_every == 0:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
        }, checkpoint_path)
        print(f"Saved checkpoint at step {step}")