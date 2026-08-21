import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from model.config import DarkGLMConfig
from model.model import DarkGLM

config = DarkGLMConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = DarkGLM(config).to(device)
model.gradient_checkpointing = True

# Continue from the Wikipedia phase - the most recent broad-knowledge checkpoint.
wikipedia_checkpoint = "checkpoints/wikipedia_best_model.pt"
model.load_state_dict(torch.load(wikipedia_checkpoint, map_location=device))
print(f"Loaded Wikipedia-trained weights from {wikipedia_checkpoint}")

# mmap_mode keeps these on disk - the two train arrays are ~1.7GB each and would
# otherwise sit in RAM alongside everything else.
train_ids = np.load("data/combined/train_ids.npy", mmap_mode="r")
train_labels = np.load("data/combined/train_labels.npy", mmap_mode="r")
val_ids = np.load("data/combined/validation_ids.npy", mmap_mode="r")
val_labels = np.load("data/combined/validation_labels.npy", mmap_mode="r")
print(f"Train examples: {len(train_ids):,}  Validation examples: {len(val_ids):,}")

def get_batch(split, batch_size):
    ids_source = train_ids if split == "train" else val_ids
    labels_source = train_labels if split == "train" else val_labels

    # One row per example here, unlike the earlier phases which sliced random
    # offsets out of a flat token stream - rows must stay intact so the answer
    # boundary the labels encode still lines up.
    rows = torch.randint(len(ids_source), (batch_size,)).numpy()
    ids_batch = np.asarray(ids_source[rows], dtype=np.int64)
    labels_batch = np.asarray(labels_source[rows], dtype=np.int64)

    # Standard causal shift: predict position i+1 from everything up to i.
    x = torch.from_numpy(ids_batch[:, :-1]).to(device)
    y = torch.from_numpy(labels_batch[:, 1:]).to(device)
    return x, y

eval_every = 250
eval_iters = 50

best_val_loss = float("inf")
# Last run quit at step 15,000 of 66,512 - only 240,000 of 532,119 examples, well
# under one pass. 40 checks = 10,000 steps, about 30% of an epoch, so a long flat
# stretch no longer ends the run before the data has been seen.
patience = 40
patience_counter = 0

@torch.no_grad()
def estimate_val_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch("validation", batch_size)
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size), y.reshape(-1), ignore_index=-100
            )
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()

# Same rate as the other narrow fine-tune phases, lower than the broad ones.
learning_rate = 3e-5
min_learning_rate = 3e-6
warmup_steps = 500
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.amp.GradScaler(device)

batch_size = 16
grad_accum_steps = 4

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
save_every = 500

start_step = 0
checkpoint_path = os.path.join(checkpoint_dir, "combined_latest.pt")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate  # override the rate baked into the checkpoint
    start_step = checkpoint["step"] + 1
    best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
    patience_counter = checkpoint.get("patience_counter", patience_counter)
    print(f"Resumed combined phase from checkpoint at step {start_step}")

# One pass over the data. The previous run scheduled the cosine over 2 epochs but
# stopped at 15,000 steps, so the rate only fell 3.0e-5 -> 2.7e-5 and the decay
# never really happened. Scheduling it over one epoch means it completes.
num_epochs = 1
steps_per_epoch = len(train_ids) // batch_size
max_steps = num_epochs * steps_per_epoch
log_every = 100

print(f"Training for {num_epochs} epochs ({steps_per_epoch:,} steps/epoch, {max_steps:,} steps total)")

def learning_rate_at(step):
    # A constant rate kept overshooting late on - the previous run went flat and
    # early-stopped at 22,750 of 71,760 with the loss bouncing, never settling.
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_learning_rate + (learning_rate - min_learning_rate) * cosine

model.train()
optimizer.zero_grad()

for step in range(start_step, max_steps):
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate_at(step)

    x, y = get_batch("train", batch_size)

    with torch.autocast(device_type=device, dtype=torch.float16):
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size), y.reshape(-1), ignore_index=-100
        )
        loss = loss / grad_accum_steps

    scaler.scale(loss).backward()

    if (step + 1) % grad_accum_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    if step % log_every == 0:
        print(f"Step {step}/{max_steps} | Loss: {loss.item() * grad_accum_steps:.4f} | lr {learning_rate_at(step):.2e}")

    if step % eval_every == 0:
        val_loss = estimate_val_loss()
        print(f"Step {step}/{max_steps} | Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/combined_best_model.pt")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping: no improvement for too long.")
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

print("Training complete.")
