from dataclasses import dataclass

@dataclass
class DarkGLMConfig:
    vocab_size: int = 32000
    embedding_dim: int = 640
    num_layers: int = 12
    num_heads: int = 8
    context_length: int = 768
    tie_weights: bool = True

    # v2 architecture switches. Defaults are v1 so every existing checkpoint still loads
    # and the smoke test can build both architectures from the same code.
    use_rope: bool = False           # rotary positions instead of a learned table
    pre_norm: bool = False           # normalise before the residual add, not after
    scale_residual_init: bool = False  # shrink residual-writing layers by 1/sqrt(2N)
    rope_base: float = 10000.0


def v2_config(**overrides):
    """All three v2 switches at once - turning on only some of them invalidates the comparison."""
    return DarkGLMConfig(use_rope=True, pre_norm=True, scale_residual_init=True, **overrides)