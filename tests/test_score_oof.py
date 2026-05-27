from scripts.score_oof import overall


def test_overall_formula():
    assert abs(overall(0.3, 0.2, 0.6) - (0.4 * 0.3 + 0.4 * 0.2 + 0.2 * 0.6)) < 1e-9
