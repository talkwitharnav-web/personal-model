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

# Continue from the best DailyDialog checkpoint, not random weights
dailydialog_checkpoint = "checkpoints/dailydialog_best_model.pt"
model.load_state_dict(torch.load(dailydialog_checkpoint, map_location=device))
print(f"Loaded DailyDialog-trained weights from {dailydialog_checkpoint}")

train_data = np.memmap("data/strategyqa/train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("data/strategyqa/validation.bin", dtype=np.uint16, mode="r")

def get_batch(split, batch_size):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - config.context_length, (batch_size,))

    x = torch.stack([torch.from_numpy(data[i:i + config.context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + config.context_length].astype(np.int64)) for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y

eval_every = 50
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

# Lower learning rate again - fine-tuning a small, narrow dataset on top of DailyDialog
learning_rate = 3e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.amp.GradScaler(device)

batch_size = 16

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
save_every = 100

start_step = 0
checkpoint_path = os.path.join(checkpoint_dir, "strategyqa_latest.pt")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate  # override rate baked into the resumed checkpoint
    start_step = checkpoint["step"] + 1
    print(f"Resumed StrategyQA phase from checkpoint at step {start_step}")

grad_accum_steps = 4
# StrategyQA is very small (~2K examples total), so many epochs are needed
num_epochs = 20
steps_per_epoch = max(1, len(train_data) // (batch_size * config.context_length))
max_steps = num_epochs * steps_per_epoch
log_every = 20

print(f"Training for {num_epochs} epochs ({steps_per_epoch} steps/epoch, {max_steps} steps total)")

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
            torch.save(model.state_dict(), "checkpoints/strategyqa_best_model.pt")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping: no improvement for too long.")
                break

    if step % save_every == 0:
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
