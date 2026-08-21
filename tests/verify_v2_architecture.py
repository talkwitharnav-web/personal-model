"""Verification for the v2 architecture switches. Run before the smoke test.

The smoke test answers "is the new architecture better". This answers the cheaper and more
important question first: "is it what we think it is". A subtly wrong RoPE still trains.
"""
import torch
import torch.nn.functional as F

from model.attention import apply_rope, build_rope_cache
from model.config import DarkGLMConfig, v2_config
from model.model import DarkGLM

V1_EXPECTED = 80_055_040
POSITION_TABLE = 768 * 640
CHECKPOINT = "checkpoints/combined_best_model.pt"

passed, failed = [], []


def check(name, condition, detail=""):
    (passed if condition else failed).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def rope_maths():
    section("RoPE maths (the part that fails silently)")
    head_dim = 640 // 8
    cos, sin = build_rope_cache(256, head_dim, 10000.0)
    check("cache shape is (seq, head_dim)", tuple(cos.shape) == (256, head_dim), str(tuple(cos.shape)))

    q = torch.randn(1, 1, 1, head_dim, dtype=torch.float32)
    k = torch.randn(1, 1, 1, head_dim, dtype=torch.float32)

    # A rotation cannot change a vector's length.
    rotated = apply_rope(q, cos[5:6], sin[5:6])
    before, after = q.norm().item(), rotated.norm().item()
    check("rotation preserves vector length", abs(before - after) < 1e-4, f"{before:.5f} -> {after:.5f}")

    def dot(m, n):
        qm = apply_rope(q, cos[m:m + 1], sin[m:m + 1])
        kn = apply_rope(k, cos[n:n + 1], sin[n:n + 1])
        return (qm * kn).sum().item()

    # THE defining property: the score depends only on the gap between positions.
    near, far = dot(2, 7), dot(100, 105)
    check("same gap (-5) gives the same score at different absolute positions",
          abs(near - far) < 1e-3, f"pos 2/7 = {near:.6f}   pos 100/105 = {far:.6f}")

    other = dot(2, 20)
    check("a different gap gives a different score (not degenerate)",
          abs(near - other) > 1e-3, f"gap -5 = {near:.6f}   gap -18 = {other:.6f}")

    at_zero = apply_rope(q, cos[0:1], sin[0:1])
    check("position 0 is the identity rotation", torch.allclose(q, at_zero, atol=1e-6))


def v1_untouched():
    section("v1 must still work (chat.py depends on it)")
    model = DarkGLM(DarkGLMConfig())
    total = sum(p.numel() for p in model.parameters())
    check("v1 parameter count unchanged", total == V1_EXPECTED, f"{total:,}")
    check("v1 still has a learned position table", model.position_embedding is not None)

    try:
        state = torch.load(CHECKPOINT, map_location="cpu")
        model.load_state_dict(state, strict=True)
        check("trained v1 checkpoint loads with strict=True", True, CHECKPOINT)
    except Exception as error:
        check("trained v1 checkpoint loads with strict=True", False, str(error)[:120])

    model.eval()
    ids = torch.randint(0, 32000, (1, 32), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = model(ids)
    check("v1 forward runs and is finite", torch.isfinite(out).all().item(), str(tuple(out.shape)))


def v2_shape_and_birth():
    section("v2 birth check (smoke test A)")
    config = v2_config()
    check("v2_config turns on all three switches",
          config.use_rope and config.pre_norm and config.scale_residual_init)

    model = DarkGLM(config)
    total = sum(p.numel() for p in model.parameters())
    expected = V1_EXPECTED - POSITION_TABLE
    check("v2 parameter count is v1 minus the position table",
          total == expected, f"{total:,} (expected {expected:,})")
    check("v2 has no learned position table", model.position_embedding is None)
    check("rope cache is not in the checkpoint",
          "rope_cos" not in model.state_dict() and "position_embedding.weight" not in model.state_dict())
    check("rope cache exists as a buffer, built once",
          hasattr(model, "rope_cos") and model.rope_cos.shape[0] == config.context_length)

    model.eval()
    ids = torch.randint(0, 32000, (2, 64), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        logits = model(ids)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, config.vocab_size), ids[:, 1:].reshape(-1))
    baseline = torch.tensor(float(config.vocab_size)).log().item()
    check("untrained loss is near ln(vocab)", abs(loss.item() - baseline) < 0.5,
          f"{loss.item():.4f} vs ln(32000)={baseline:.4f}")
    check("no NaN in the forward pass", torch.isfinite(logits).all().item())


def switches_do_something():
    section("the switches actually change behaviour")
    seed = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 32000, (1, 16), generator=seed)

    def logits_for(config):
        torch.manual_seed(7)
        model = DarkGLM(config).eval()
        with torch.no_grad():
            return model(ids)

    post = logits_for(DarkGLMConfig())
    pre = logits_for(DarkGLMConfig(pre_norm=True))
    check("pre_norm changes the output", not torch.allclose(post, pre, atol=1e-5))

    torch.manual_seed(7)
    plain = DarkGLM(DarkGLMConfig())
    torch.manual_seed(7)
    scaled = DarkGLM(DarkGLMConfig(scale_residual_init=True))
    factor = (2 * 12) ** -0.5
    a = plain.blocks[0].attention.output_proj.weight
    b = scaled.blocks[0].attention.output_proj.weight
    check("residual scaling shrinks the attention output projection",
          torch.allclose(a * factor, b, atol=1e-7), f"factor {factor:.4f}")
    c = plain.blocks[0].feed_forward[2].weight
    d = scaled.blocks[0].feed_forward[2].weight
    check("residual scaling shrinks the MLP output projection", torch.allclose(c * factor, d, atol=1e-7))
    check("residual scaling leaves other layers alone",
          torch.allclose(plain.blocks[0].feed_forward[0].weight,
                         scaled.blocks[0].feed_forward[0].weight, atol=1e-7))


def training_path():
    section("v2 survives the real training path")
    model = DarkGLM(v2_config())
    model.gradient_checkpointing = True
    model.train()
    ids = torch.randint(0, 32000, (2, 48), generator=torch.Generator().manual_seed(3))
    logits = model(ids)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 32000), ids[:, 1:].reshape(-1))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check("gradient checkpointing forward+backward works", len(grads) > 0, f"{len(grads)} tensors with grad")
    check("no NaN or inf in any gradient", all(torch.isfinite(g).all().item() for g in grads))
    embed_grad = model.token_embedding.weight.grad
    check("gradient reaches the token embeddings", embed_grad is not None and embed_grad.abs().sum() > 0)

    short = model(torch.randint(0, 32000, (1, 8)))
    check("a shorter sequence still works (rope cache slices correctly)",
          torch.isfinite(short).all().item(), str(tuple(short.shape)))


def main():
    rope_maths()
    v1_untouched()
    v2_shape_and_birth()
    switches_do_something()
    training_path()
    print(f"\n{len(passed)} passed, {len(failed)} failed")
    if failed:
        print("FAILED: " + ", ".join(failed))


if __name__ == "__main__":
    main()
