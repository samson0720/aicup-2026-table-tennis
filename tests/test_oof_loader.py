import numpy as np
import pandas as pd

from scripts.oof_loader import write_oof, read_oof, average_over_seeds, TARGET_CLASS_COUNTS


def test_roundtrip_action(tmp_path, monkeypatch):
    import scripts.oof_loader as ol
    monkeypatch.setattr(ol, "OOF_DIR", tmp_path)
    n = 50
    probs = np.random.default_rng(0).dirichlet(np.ones(19), size=n).astype(np.float32)
    write_oof(
        "dummy", "action",
        rally_uid=np.arange(n),
        seed=np.full(n, 11),
        fold=np.arange(n) % 5,
        cut=np.full(n, 3),
        probs=probs,
    )
    df = read_oof("dummy", "action")
    assert len(df) == n
    p_cols = [f"p_{i}" for i in range(19)]
    assert all(c in df.columns for c in p_cols)
    s = df[p_cols].sum(axis=1).to_numpy()
    assert np.allclose(s, 1.0, atol=1e-3)


def test_roundtrip_server(tmp_path, monkeypatch):
    import scripts.oof_loader as ol
    monkeypatch.setattr(ol, "OOF_DIR", tmp_path)
    n = 30
    probs = np.random.default_rng(1).random((n, 1)).astype(np.float32)
    write_oof(
        "dummy", "server",
        rally_uid=np.arange(n),
        seed=np.full(n, 11),
        fold=np.arange(n) % 5,
        cut=np.full(n, 3),
        probs=probs,
    )
    df = read_oof("dummy", "server")
    assert len(df) == n
    assert "p_1" in df.columns
    assert df["p_1"].between(0.0, 1.0).all()


def test_average_over_seeds(tmp_path, monkeypatch):
    import scripts.oof_loader as ol
    monkeypatch.setattr(ol, "OOF_DIR", tmp_path)
    rng = np.random.default_rng(0)
    rally = np.repeat(np.arange(10), 5)  # 10 rallies × 5 seeds
    seeds = np.tile([11, 22, 33, 44, 55], 10)
    probs = rng.dirichlet(np.ones(10), size=50).astype(np.float32)
    write_oof(
        "dummy", "point",
        rally_uid=rally, seed=seeds, fold=np.zeros(50, dtype=np.int32),
        cut=np.full(50, 3), probs=probs,
    )
    df = read_oof("dummy", "point")
    avg = average_over_seeds(df, "point")
    assert len(avg) == 10
    assert set(avg["rally_uid"]) == set(range(10))
    # average is row-wise mean across 5 seeds, must still sum to ~1.
    p_cols = [f"p_{i}" for i in range(10)]
    s = avg[p_cols].sum(axis=1).to_numpy()
    assert np.allclose(s, 1.0, atol=1e-3)


def test_target_class_counts_constant():
    assert TARGET_CLASS_COUNTS == {"action": 19, "point": 10, "server": 1}
