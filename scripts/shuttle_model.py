"""ShuttleNet-style dual-encoder forecaster (v4 L1). Rally-progress encoder +
player-style encoder + position-aware gated fusion. action + point heads only.
Reimplemented in pure PyTorch (no new deps), inspired by ShuttleNet (AAAI'22)."""
from __future__ import annotations

import torch
from torch import nn

from scripts.seq_dataset import CATEGORICAL_COLS, MAX_LEN, SCORE_FLOAT_COLS
from scripts.seq_model import DEFAULT_CARDINALITIES

PLAYER_COLS = ("gamePlayerId", "gamePlayerOtherId", "sex")


def _encoder(d_model, nhead, num_layers, dim_feedforward, dropout):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
        dropout=dropout, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=num_layers)


class ShuttleForecaster(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=512,
                 dropout=0.3, cardinalities=None):
        super().__init__()
        cards = cardinalities or DEFAULT_CARDINALITIES
        emb_dim = d_model // len(CATEGORICAL_COLS)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cards[c], emb_dim, padding_idx=0) for c in CATEGORICAL_COLS])
        self.cat_cols = CATEGORICAL_COLS
        self.player_idx = [CATEGORICAL_COLS.index(c) for c in PLAYER_COLS]
        cat_dim = emb_dim * len(CATEGORICAL_COLS)
        self.rally_proj = nn.Linear(cat_dim + len(SCORE_FLOAT_COLS), d_model)
        self.style_proj = nn.Linear(emb_dim * len(PLAYER_COLS), d_model)
        self.pos = nn.Embedding(MAX_LEN, d_model)
        self.rally_enc = _encoder(d_model, nhead, num_layers, dim_feedforward, dropout)
        self.style_enc = _encoder(d_model, nhead, max(1, num_layers // 2), dim_feedforward, dropout)
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, 19)
        self.point_head = nn.Linear(d_model, 10)

    def _embed(self, tokens):
        return [tbl(tokens[:, :, i].clamp(0, tbl.num_embeddings - 1))
                for i, tbl in enumerate(self.embeddings)]

    def _contexts(self, tokens, floats, mask):
        emb = self._embed(tokens)
        rally_in = torch.cat([*emb, floats], dim=-1)
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        rally = self.rally_enc(self.rally_proj(rally_in) + self.pos(positions),
                               src_key_padding_mask=~mask)
        style_in = torch.cat([emb[i] for i in self.player_idx], dim=-1)
        style = self.style_enc(self.style_proj(style_in) + self.pos(positions),
                               src_key_padding_mask=~mask)
        lengths = mask.long().sum(1).clamp(min=1) - 1
        ar = torch.arange(rally.shape[0], device=rally.device)
        return rally[ar, lengths], style[ar, lengths]

    def last_gate(self, tokens, floats, mask):
        r, s = self._contexts(tokens, floats, mask)
        return self.gate(torch.cat([r, s], dim=-1))

    def forward(self, tokens, floats, mask):
        r, s = self._contexts(tokens, floats, mask)
        g = self.gate(torch.cat([r, s], dim=-1))
        fused = self.norm(g * r + (1.0 - g) * s)
        return {"action_logits": self.action_head(fused),
                "point_logits": self.point_head(fused)}
