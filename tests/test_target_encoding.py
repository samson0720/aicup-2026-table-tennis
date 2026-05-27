import numpy as np
import pandas as pd

from scripts.target_encoding import SmoothedEncoder, build_player_encoders


def test_smoothing_converges_to_global_on_empty_history():
    train = pd.DataFrame({"player": [], "y": []})
    enc = SmoothedEncoder(
        keys=["player"],
        n_classes=3,
        alpha=20.0,
        global_prior=np.array([0.5, 0.3, 0.2]),
    )
    enc.fit(train, y_col="y")
    out = enc.transform(pd.DataFrame({"player": [42, 99]}))
    np.testing.assert_allclose(out, np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]), atol=1e-7)


def test_smoothing_blends_with_alpha():
    train = pd.DataFrame({"player": [1, 1, 1, 1], "y": [0, 0, 1, 2]})
    enc = SmoothedEncoder(
        keys=["player"],
        n_classes=3,
        alpha=4.0,
        global_prior=np.array([0.50, 0.25, 0.25]),
    ).fit(train, y_col="y")
    out = enc.transform(pd.DataFrame({"player": [1]}))
    np.testing.assert_allclose(out[0], [0.5, 0.25, 0.25], atol=1e-7)


def test_oof_no_self_inclusion():
    train = pd.DataFrame({"player": [1, 1, 2, 2, 2], "y": [0, 0, 1, 1, 2]})
    valid = pd.DataFrame({"player": [3, 3]})
    enc = SmoothedEncoder(keys=["player"], n_classes=3, alpha=10.0).fit(train, y_col="y")
    out = enc.transform(valid)
    np.testing.assert_allclose(out, np.tile(enc._global, (2, 1)), atol=1e-7)


def test_does_not_overwrite_stats_on_repeat_fit():
    train1 = pd.DataFrame({"player": [1, 1], "y": [0, 0]})
    train2 = pd.DataFrame({"player": [2, 2], "y": [1, 1]})
    enc = SmoothedEncoder(keys=["player"], n_classes=2, alpha=4.0).fit(train1, y_col="y")
    enc.fit(train2, y_col="y")
    assert enc._stats.index.tolist() == [2]
    out = enc.transform(pd.DataFrame({"player": [1, 2]}))
    np.testing.assert_allclose(out[0], enc._global, atol=1e-7)
    assert out[1, 1] > out[1, 0]


def test_multi_encoder_bundle_shape():
    train = pd.DataFrame({
        "player": [1, 1, 2, 2, 2, 3],
        "phase": [0, 0, 1, 2, 2, 2],
        "opponent": [9, 9, 8, 8, 8, 9],
        "y_action": [0, 1, 2, 0, 1, 3],
        "y_point": [0, 0, 1, 1, 2, 0],
        "y_server": [1, 0, 1, 1, 0, 1],
    })
    encs = build_player_encoders(train, n_action=4, n_point=3)
    out = encs.transform(train.head(3))
    assert out.shape[0] == 3
    assert out.shape[1] == 3 * (4 + 3 + 2)
    assert np.isfinite(out).all()
