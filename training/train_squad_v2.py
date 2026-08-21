import os
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

# Picks up where the combined instruction phase left off.
combined_checkpoint = "checkpoints/combined_best_model.pt"
model.load_state_dict(torch.load(combined_checkpoint, map_location=device))
print(f"Loaded combined weights from {combined_checkpoint}")

train_ids = np.load("data/squad_v2/train_ids.npy", mmap_mode="r")
train_labels = np.load("data/squad_v2/train_labels.npy", mmap_mode="r")
val_ids = np.load("data/squad_v2/validation_ids.npy", mmap_mode="r")
val_labels = np.load("data/squad_v2/validation_labels.npy", mmap_mode="r")
print(f"Train examples: {len(train_ids):,}  Validation examples: {len(val_ids):,}")

def get_batch(split, batch_size):
    ids_source = train_ids if split == "train" else val_ids
    labels_source = train_labels if split == "train" else val_labels

    rows = torch.randint(len(ids_source), (batch_size,)).numpy()
    ids_batch = np.asarray(ids_source[rows], dtype=np.int64)
    labels_batch = np.asarray(labels_source[rows], dtype=np.int64)

    x = torch.from_numpy(ids_batch[:, :-1]).to(device)
    y = torch.from_numpy(labels_batch[:, 1:]).to(device)
    return x, y

eval_every = 250
eval_iters = 50

best_val_loss = float("inf")
# Shorter run than the combined phase, so a shorter patience - 10 checks is
# 2,500 steps, about 15% of the whole run.
patience = 10
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

# Deliberately lower than the combined phase's 3e-5 - this is a narrow behaviour
# tweak (learning to decline), not a knowledge phase, so it should nudge the
# weights rather than reshape them.
learning_rate = 1e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.amp.GradScaler(device)

batch_size = 16
grad_accum_steps = 4

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
save_every = 500

start_step = 0
checkpoint_path = os.path.join(checkpoint_dir, "squad_v2_latest.pt")
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
    print(f"Resumed SQuAD v2 phase from checkpoint at step {start_step}")

num_epochs = 2
steps_per_epoch = len(train_ids) // batch_size
max_steps = num_epochs * steps_per_epoch
log_every = 100

print(f"Training for {num_epochs} epochs ({steps_per_epoch:,} steps/epoch, {max_steps:,} steps total)")

model.train()
optimizer.zero_grad()

for step in range(start_step, max_steps):
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
        print(f"Step {step}/{max_steps} | Loss: {loss.item() * grad_accum_steps:.4f}")

    if step % eval_every == 0:
        val_loss = estimate_val_loss()
        print(f"Step {step}/{max_steps} | Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/squad_v2_best_model.pt")
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
