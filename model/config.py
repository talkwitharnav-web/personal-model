from dataclasses import dataclass

@dataclass
class DarkGLMConfig:
    vocab_size: int = 32000
    embedding_dim: int = 640
    num_layers: int = 12
    num_heads: int = 8
    context_length: int = 768
    tie_weights: bool = True