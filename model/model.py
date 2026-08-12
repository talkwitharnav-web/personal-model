import torch
import torch.nn as nn
import torch.utils.checkpoint
from model.config import DarkGLMConfig
from model.transformer_block import TransformerBlock

class DarkGLM(nn.Module):
    def __init__(self, config: DarkGLMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.output_layer = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.output_layer.weight = self.token_embedding.weight
        self.gradient_checkpointing = False

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        token_embeds = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device)
        position_embeds = self.position_embedding(positions)

        x = token_embeds + position_embeds

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.final_norm(x)
        logits = self.output_layer(x)
        return logits

if __name__ == "__main__":
    config = DarkGLMConfig()
    model = DarkGLM(config)

    fake_input = torch.randint(0, config.vocab_size, (2, 10))
    logits = model(fake_input)

    print("Input shape:", fake_input.shape)
    print("Output shape:", logits.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")