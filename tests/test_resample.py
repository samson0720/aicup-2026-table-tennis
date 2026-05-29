import numpy as np
import pandas as pd

from scripts.resample import oversample_rare


def test_oversample_brings_rare_classes_up_to_floor():
    # class 0: 100 rows, class 1: 10 rows, class 2: 5 rows
    y = pd.Series([0] * 100 + [1] * 10 + [2] * 5)
    x = pd.DataFrame({"f": np.arange(len(y))})
    xo, yo = oversample_rare(x, y, seed=0, min_frac=0.5)
    counts = yo.value_counts().to_dict()
    # rare classes raised to >= 0.5 * max (=50); majority untouched
    assert counts[0] == 100
    assert counts[1] >= 50
    assert counts[2] >= 50
    # only duplicates real rows (feature values stay within the original set)
    assert set(xo["f"]).issubset(set(x["f"]))
    # x and y stay aligned (same length, index reset)
    assert len(xo) == len(yo)


def test_oversample_noop_when_balanced():
    y = pd.Series([0] * 50 + [1] * 50)
    x = pd.DataFrame({"f": np.arange(100)})
    xo, yo = oversample_rare(x, y, seed=0, min_frac=0.5)
    assert len(xo) == 100  # already balanced -> no change
