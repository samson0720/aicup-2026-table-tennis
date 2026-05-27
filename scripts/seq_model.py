"""Small multi-task Transformer for rally prefixes."""
from __future__ import annotations

import torch
from torch import nn

from scripts.seq_dataset import CATEGORICAL_COLS, MAX_LEN, SCORE_FLOAT_COLS


DEFAULT_CARDINALITIES = {
    "strikeId": 8,
    "handId": 8,
    "strengthId": 8,
    "spinId": 16,
    "pointId": 16,
    "actionId": 32,
    "positionId": 8,
    "gamePlayerId": 512,
    "gamePlayerOtherId": 512,
    "sex": 4,
}


class RallyTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        cardinalities: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        cards = cardinalities or DEFAULT_CARDINALITIES
        emb_dim = d_model // len(CATEGORICAL_COLS)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cards[col], emb_dim, padding_idx=0) for col in CATEGORICAL_COLS]
        )
        cat_dim = emb_dim * len(CATEGORICAL_COLS)
        self.input_proj = nn.Linear(cat_dim + len(SCORE_FLOAT_COLS), d_model)
        self.pos = nn.Embedding(MAX_LEN, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, 19)
        self.point_head = nn.Linear(d_model, 10)
        self.server_head = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor, floats: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        emb = []
        for i, table in enumerate(self.embeddings):
            ids = tokens[:, :, i].clamp(min=0, max=table.num_embeddings - 1)
            emb.append(table(ids))
        x = torch.cat([*emb, floats], dim=-1)
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        x = self.input_proj(x) + self.pos(positions)
        encoded = self.encoder(x, src_key_padding_mask=~mask)

        lengths = mask.long().sum(dim=1).clamp(min=1) - 1
        pooled = encoded[torch.arange(encoded.shape[0], device=encoded.device), lengths]
        pooled = self.norm(pooled)
        return {
            "action_logits": self.action_head(pooled),
            "point_logits": self.point_head(pooled),
            "server_logit": self.server_head(pooled).squeeze(-1),
        }
