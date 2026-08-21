import torch
import torch.nn as nn
from model.config import DarkGLMConfig
from model.attention import CausalSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, config: DarkGLMConfig):
        super().__init__()
        self.pre_norm = config.pre_norm
        self.attention = CausalSelfAttention(config)
        self.norm1 = nn.LayerNorm(config.embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim *4),
            nn.GELU(),
            nn.Linear(config.embedding_dim * 4, config.embedding_dim),
        )
        self.norm2 = nn.LayerNorm(config.embedding_dim)

    def forward(self, x, cos=None, sin=None):
        # Pre-norm keeps the residual path free of LayerNorms, so the backward signal reaches
        # the early layers unsqueezed. Same weights either way - only the order changes.
        if self.pre_norm:
            x = x + self.attention(self.norm1(x), cos, sin)
            x = x + self.feed_forward(self.norm2(x))
            return x

        attention_output = self.attention(x, cos, sin)
        x = self.norm1(x + attention_output)

        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x+feed_forward_output)

        return x

if __name__ == "__main__":
    config = DarkGLMConfig()
    block = TransformerBlock(config)

    fake_input = torch.randn(2, 10, config.embedding_dim)
    output = block(fake_input)

    print("Input shape:", fake_input.shape)
    print("Output shape:", output.shape)
