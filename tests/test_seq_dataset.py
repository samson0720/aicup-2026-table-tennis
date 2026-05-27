import pandas as pd
import torch

from scripts.seq_dataset import CATEGORICAL_COLS, MAX_LEN, RallyPrefixDataset, collate_batch


def _fake_train():
    rows = []
    for rally_uid in range(5):
        for strike in range(1, 6):
            rows.append({
                "rally_uid": rally_uid,
                "match": rally_uid // 2,
                "strikeNumber": strike,
                "scoreSelf": strike - 1,
                "scoreOther": 0,
                "gamePlayerId": (strike + rally_uid) % 4 + 1,
                "gamePlayerOtherId": (strike + rally_uid + 1) % 4 + 1,
                "strikeId": 1,
                "handId": 1,
                "strengthId": 1,
                "spinId": 1,
                "pointId": (strike + rally_uid) % 10,
                "actionId": (strike + rally_uid) % 19,
                "positionId": 1,
                "sex": 1,
                "numberGame": 1,
                "rally_id": rally_uid,
                "serverGetPoint": rally_uid % 2,
            })
    return pd.DataFrame(rows)


def _fake_splits():
    return pd.DataFrame({
        "rally_uid": list(range(5)),
        "match": [0, 0, 1, 1, 2],
        "seed": [11] * 5,
        "fold": [0, 0, 1, 1, 1],
        "cut_strikeNumber": [3, 4, 5, 2, 5],
        "phase_bucket": ["phase1"] * 5,
    })


def test_dataset_yields_padded_tensor():
    ds = RallyPrefixDataset(_fake_train(), _fake_splits(), seed=11)
    item = ds[0]
    assert item["tokens"].shape == (MAX_LEN, len(CATEGORICAL_COLS))
    assert item["floats"].shape[0] == MAX_LEN
    assert item["mask"].dtype == torch.bool
    assert item["y_action"].dtype == torch.long


def test_no_label_stroke_in_tokens():
    ds = RallyPrefixDataset(_fake_train(), _fake_splits(), seed=11)
    strike_col = None
    assert "strikeNumber" not in CATEGORICAL_COLS
    for item in ds:
        assert int(item["mask"].sum()) == item["target_strike"] - 1
        assert strike_col is None


def test_collate_batch_shapes():
    ds = RallyPrefixDataset(_fake_train(), _fake_splits(), seed=11, fold=1)
    batch = collate_batch([ds[0], ds[1]])
    assert batch["tokens"].shape == (2, MAX_LEN, len(CATEGORICAL_COLS))
    assert batch["y_server"].shape == (2,)
