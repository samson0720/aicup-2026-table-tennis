import numpy as np
from scripts.produce_joint_oof import fit_point_given_action, marginalize_point

def test_marginalize_is_simplex_and_matches_hand_calc():
    cond = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])  # (n_action=3, n_point=2)
    phat_a = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 2 rows
    out = marginalize_point(phat_a, cond)
    assert out.shape == (2, 2)
    assert np.allclose(out[0], [1.0, 0.0])
    assert np.allclose(out[1], [0.5, 0.5])
    assert np.allclose(out.sum(axis=1), 1.0)

def test_fit_conditional_simplex():
    import pandas as pd
    df = pd.DataFrame({"y_actionId": [0, 0, 1, 2, 2], "y_pointId": [3, 3, 4, 5, 6]})
    cond = fit_point_given_action(df, n_action=19, n_point=10, alpha=2.0)
    assert cond.shape == (19, 10)
    assert np.allclose(cond.sum(axis=1), 1.0, atol=1e-6)
