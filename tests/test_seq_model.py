import torch

from scripts.seq_dataset import CATEGORICAL_COLS, MAX_LEN, SCORE_FLOAT_COLS
from scripts.seq_model import RallyTransformer


def test_transformer_forward_shapes():
    model = RallyTransformer(d_model=64, nhead=4, num_layers=1, dim_feedforward=128, dropout=0.0)
    tokens = torch.ones(3, MAX_LEN, len(CATEGORICAL_COLS), dtype=torch.long)
    floats = torch.zeros(3, MAX_LEN, len(SCORE_FLOAT_COLS))
    mask = torch.zeros(3, MAX_LEN, dtype=torch.bool)
    mask[:, :5] = True
    out = model(tokens, floats, mask)
    assert out["action_logits"].shape == (3, 19)
    assert out["point_logits"].shape == (3, 10)
    assert out["server_logit"].shape == (3,)


def test_transformer_parameter_count_sane():
    model = RallyTransformer(d_model=64, nhead=4, num_layers=1, dim_feedforward=128)
    n_params = sum(p.numel() for p in model.parameters())
    assert 40_000 < n_params < 2_000_000
