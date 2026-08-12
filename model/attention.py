import torch
import torch.nn as nn
from model.config import DarkGLMConfig

class CausalSelfAttention(nn.Module):
    def __init__(self, config: DarkGLMConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads

        self.query_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.key_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.value_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.output_proj = nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.shape

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

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