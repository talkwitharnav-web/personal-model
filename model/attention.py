import torch
import torch.nn as nn
from model.config import DarkGLMConfig


def build_rope_cache(seq_len, head_dim, base):
    """cos/sin tables for rotary positions.

    Built ONCE by DarkGLM and passed down to every layer. Rebuilding these per block is the
    classic RoPE performance bug and shows up as a large drop in tokens/sec.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    # Duplicated so the table lines up with the half-split rotation below.
    table = torch.cat((freqs, freqs), dim=-1)
    return table.cos(), table.sin()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x, cos, sin):
    """x is (batch, heads, seq, head_dim); cos/sin are (seq, head_dim)."""
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config: DarkGLMConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.use_rope = config.use_rope

        self.query_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.key_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.value_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.output_proj = nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(self, x, cos=None, sin=None):
        batch_size, seq_len, embedding_dim = x.shape

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            if cos is None or sin is None:
                raise ValueError("use_rope is on but no rope cache reached the attention layer")
            # Queries and keys only - rotating values would corrupt the content being retrieved.
            Q = apply_rope(Q, cos, sin)
            K = apply_rope(K, cos, sin)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = attention_weights @ V

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, embedding_dim)

        output = self.output_proj(attention_output)
        return output

if __name__ == "__main__":
    config = DarkGLMConfig()
    attention = CausalSelfAttention(config)

    fake_input = torch.randn(2, 10, config.embedding_dim)
    output = attention(fake_input)

    print("Input shape:", fake_input.shape)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    config = DarkGLMConfig()
    attention = CausalSelfAttention(config)

    fake_input = torch.randn(2, 10, config.embedding_dim)
    output = attention(fake_input)

    print("Input shape:", fake_input.shape)
    print("Output shape:", output.shape)