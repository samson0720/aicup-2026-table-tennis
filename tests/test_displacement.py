"""Unit tests for the leakage-free displacement / pressure-proxy features (Idea 1)."""
import pandas as pd

from scripts.train_lgbm_baseline import (
    _coord25,
    _coord33,
    _euclid,
    add_prefix_features,
    displacement_features,
)


def _prefix(point_ids):
    """Minimal prefix frame with the columns add_prefix_features/displacement read."""
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


def test_coord_layouts():
    assert _coord33(0) == (0.0, 0.0)
    assert _coord33(8) == (2.0, 2.0)
    assert _coord33(9) == (1.0, 1.0)  # zone 9 -> bounded centre
    assert _coord25(9) == (1.0, 4.0)


def test_my_run_uses_t1_and_t3_same_frame():
    # zones at t-3=0 (0,0) and t-1=8 (2,2): euclid = sqrt(8)
    f = displacement_features(_prefix([0, 5, 8]))
    assert abs(f["disp_my_euclid33"] - _euclid((0, 0), (2, 2))) < 1e-9
    assert f["disp_my_manh33"] == 4.0


def test_incoming_always_present_and_short_prefix_sentinels():
    f = displacement_features(_prefix([4]))  # only t-1 known
    assert f["disp_incoming_row33"] == _coord33(4)[0]
    assert f["disp_my_euclid33"] == -1.0  # needs >=3
    assert f["disp_opp_euclid33"] == -1.0  # needs >=4


def test_opt_in_default_off():
    pref = _prefix([0, 1, 2, 3])
    base = add_prefix_features(pref, 5)
    with_d = add_prefix_features(pref, 5, with_displacement=True)
    assert not any(k.startswith("disp_") for k in base)
    assert any(k.startswith("disp_") for k in with_d)
    # adding displacement must not change any pre-existing feature
    assert all(with_d[k] == base[k] for k in base)
