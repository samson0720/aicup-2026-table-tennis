"""Compact FT-Transformer (numerical feature tokenizer) for the tabular prefix
features. Each standardized feature becomes a token; a CLS token is pooled into
multi-task action/point/server heads. Pure PyTorch (no new deps)."""
from __future__ import annotations

import torch
from torch import nn


class FTTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 8,
                 num_layers: int = 3, dim_feedforward: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, 19)
        self.point_head = nn.Linear(d_model, 10)
        self.server_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B, n_features) standardized
        tok = x.unsqueeze(-1) * self.weight + self.bias  # (B, n_features, d_model)
        cls = self.cls.expand(x.shape[0], -1, -1)
        h = torch.cat([cls, tok], dim=1)
        h = self.encoder(h)
        pooled = self.norm(h[:, 0])
        return {
            "action_logits": self.action_head(pooled),
            "point_logits": self.point_head(pooled),
            "server_logit": self.server_head(pooled).squeeze(-1),
        }
