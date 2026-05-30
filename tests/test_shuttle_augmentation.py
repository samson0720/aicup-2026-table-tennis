import torch

from scripts.seq_dataset import (
    CATEGORICAL_COLS,
    MirrorPositionActionDataset,
    collate_batch,
)
from scripts.train_shuttle import _multitask_loss


def _item(positions):
    tokens = torch.zeros((len(positions), len(CATEGORICAL_COLS)), dtype=torch.long)
    tokens[:, CATEGORICAL_COLS.index("positionId")] = torch.tensor(positions)
    return {
        "tokens": tokens,
        "floats": torch.zeros((len(positions), 2)),
        "mask": torch.ones(len(positions), dtype=torch.bool),
        "y_action": torch.tensor(4),
        "y_point": torch.tensor(6),
        "y_server": torch.tensor(1.0),
        "rally_uid": 123,
        "fold": 0,
        "target_strike": 5,
    }


def test_mirror_position_action_dataset_doubles_and_mirrors_only_left_right():
    base = [_item([0, 1, 2, 3])]
    ds = MirrorPositionActionDataset(base)

    assert len(ds) == 2
    original = ds[0]
    mirrored = ds[1]
    pos_idx = CATEGORICAL_COLS.index("positionId")
    assert original["tokens"][:, pos_idx].tolist() == [0, 1, 2, 3]
    assert mirrored["tokens"][:, pos_idx].tolist() == [0, 1, 3, 2]
    assert original["point_loss_weight"].item() == 1.0
    assert mirrored["point_loss_weight"].item() == 0.0
    assert mirrored["y_action"].item() == original["y_action"].item()
    assert mirrored["y_point"].item() == original["y_point"].item()


def test_multitask_loss_ignores_mirrored_point_label():
    ds = MirrorPositionActionDataset([_item([2])])
    batch = collate_batch([ds[0], ds[1]])
    out = {
        "action_logits": torch.zeros((2, 19)),
        "point_logits": torch.zeros((2, 10)),
    }
    out["point_logits"][1, 6] = -100.0
    weights_action = torch.ones(19)
    weights_point = torch.ones(10)

    loss = _multitask_loss(out, batch, torch.device("cpu"), weights_action, weights_point)
    expected = torch.log(torch.tensor(19.0)) + torch.log(torch.tensor(10.0))
    assert torch.isclose(loss, expected)
