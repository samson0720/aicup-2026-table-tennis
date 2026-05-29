import numpy as np
from scripts.produce_markov2_oof import fit_tables2, predict2

def _toy():
    import pandas as pd
    return pd.DataFrame({
        "y_actionId": [0, 0, 1, 1, 2, 0, 1, 2],
        "last1_actionId": [5, 5, 6, 6, 5, 5, 6, 5],
        "last2_actionId": [3, 3, 4, 4, 3, 3, 4, 3],
        "next_gamePlayerId_inferred": [10, 10, 11, 11, 10, 10, 11, 10],
    })

def test_fit_predict_shape_and_simplex():
    df = _toy()
    tables = fit_tables2(df, "action")
    p = predict2(df, "action", tables)
    assert p.shape == (len(df), 19)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert (p >= 0).all()

def test_backoff_unseen_context_falls_back_to_global():
    df = _toy()
    tables = fit_tables2(df, "action")
    import pandas as pd
    unseen = pd.DataFrame({
        "y_actionId": [0], "last1_actionId": [99],
        "last2_actionId": [99], "next_gamePlayerId_inferred": [999],
    })
    p = predict2(unseen, "action", tables)
    glob = tables[0]
    assert np.allclose(p[0], glob, atol=1e-6)
