"""Test the seq2 test-inference dataset path (no labels, cut = max+1)."""
import pandas as pd

from scripts.seq_dataset import CATEGORICAL_COLS, SCORE_FLOAT_COLS
from scripts.train_seq_pilot import build_test_dataset


def _fake_test() -> pd.DataFrame:
    rows = []
    # rally 100: 3 observed strokes; rally 200: 1 observed stroke
    for rally_uid, n in [(100, 3), (200, 1)]:
        for sn in range(1, n + 1):
            row = {"rally_uid": rally_uid, "strikeNumber": sn}
            for col in CATEGORICAL_COLS:
                row[col] = 1
            for col in SCORE_FLOAT_COLS:
                row[col] = 0.0
            rows.append(row)
    # deliberately omit serverGetPoint — it is absent in test_new.csv
    return pd.DataFrame(rows)


def test_build_test_dataset_prefix_is_all_observed_strokes():
    ds = build_test_dataset(_fake_test(), seed=11)
    assert len(ds) == 2

    by_uid = {ds[i]["rally_uid"]: ds[i] for i in range(len(ds))}
    # rally 100: prefix = all 3 observed strokes, target cut = max+1 = 4
    assert int(by_uid[100]["mask"].sum()) == 3
    assert int(by_uid[100]["target_strike"]) == 4
    # rally 200: prefix = the single observed stroke, target cut = 2
    assert int(by_uid[200]["mask"].sum()) == 1
    assert int(by_uid[200]["target_strike"]) == 2
