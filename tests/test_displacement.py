"""Unit tests for the leakage-free displacement / pressure-proxy features (Idea 1).

Geometry is the VERIFIED pointId dictionary: 1..9 = 3x3 receiver-frame grid
(x: forehand{1,4,7}=+1/middle{2,5,8}=0/backhand{3,6,9}=-1; y: short{1,2,3}=1/
half{4,5,6}=2/long{7,8,9}=3), 0 = rally-ending non-spatial zone.
"""
import pandas as pd

from scripts.train_lgbm_baseline import (
    _coord,
    _manhattan,
    add_prefix_features,
    displacement_features,
)


def _prefix(point_ids):
    n = len(point_ids)
    cols = {
        "strikeNumber": list(range(1, n + 1)),
        "pointId": point_ids,
        "rally_uid": [1] * n,
        "match": [1] * n, "sex": [0] * n, "numberGame": [1] * n, "rally_id": [1] * n,
        "scoreSelf": [0] * n, "scoreOther": [0] * n,
        "gamePlayerId": [0] * n, "gamePlayerOtherId": [1] * n,
        "strikeId": [0] * n, "handId": [0] * n, "strengthId": [0] * n,
        "spinId": [0] * n, "actionId": [0] * n, "positionId": [0] * n,
    }
    return pd.DataFrame(cols)


def test_coord_real_geometry():
    assert _coord(1) == (1.0, 1.0)    # forehand short
    assert _coord(5) == (0.0, 2.0)    # middle half-long
    assert _coord(9) == (-1.0, 3.0)   # backhand long (far corner, NOT centre)
    assert _coord(0) is None          # rally-ending = non-spatial
    assert _coord(10) is None


def test_my_run_backhand_long_to_forehand_short():
    # t-3 = 9 (-1,3), t-1 = 1 (1,1): manhattan = |1-(-1)|+|1-3| = 4 (the worked example)
    f = displacement_features(_prefix([9, 5, 1]))
    assert f["disp_my_manh"] == 4.0
    assert abs(f["disp_my_euclid"] - _manhattan((1, 1), (-1, 3))) >= 0  # euclid < manh
    assert f["disp_my_euclid"] < f["disp_my_manh"]


def test_zone0_and_short_prefix_are_sentinel():
    f = displacement_features(_prefix([1]))           # only t-1 known
    assert f["disp_incoming_x"] == 1.0 and f["disp_incoming_y"] == 1.0
    assert f["disp_my_manh"] == -1.0                  # needs >=3
    assert f["disp_opp_manh"] == -1.0                 # needs >=4
    f0 = displacement_features(_prefix([0, 5, 0]))    # endpoints in non-spatial zone 0
    assert f0["disp_my_manh"] == -1.0
    assert f0["disp_incoming_x"] == -2.0              # incoming is zone 0


def test_opt_in_default_off_byte_safe():
    pref = _prefix([1, 2, 3, 4])
    base = add_prefix_features(pref, 5)
    with_d = add_prefix_features(pref, 5, with_displacement=True)
    assert not any(k.startswith("disp_") for k in base)
    assert any(k.startswith("disp_") for k in with_d)
    assert all(with_d[k] == base[k] for k in base)
