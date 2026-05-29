import torch
from scripts.shuttle_model import ShuttleForecaster
from scripts.seq_dataset import CATEGORICAL_COLS, SCORE_FLOAT_COLS, MAX_LEN

def test_forward_shapes():
    b = 4
    model = ShuttleForecaster(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    tokens = torch.randint(0, 7, (b, MAX_LEN, len(CATEGORICAL_COLS)))
    floats = torch.randn(b, MAX_LEN, len(SCORE_FLOAT_COLS))
    mask = torch.ones(b, MAX_LEN, dtype=torch.bool)
    out = model(tokens, floats, mask)
    assert out["action_logits"].shape == (b, 19)
    assert out["point_logits"].shape == (b, 10)
    assert "server_logit" not in out

def test_gate_is_bounded():
    model = ShuttleForecaster(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    b = 2
    tokens = torch.zeros(b, MAX_LEN, len(CATEGORICAL_COLS), dtype=torch.long)
    floats = torch.zeros(b, MAX_LEN, len(SCORE_FLOAT_COLS))
    mask = torch.ones(b, MAX_LEN, dtype=torch.bool)
    g = model.last_gate(tokens, floats, mask)
    assert g.min() >= 0.0 and g.max() <= 1.0
