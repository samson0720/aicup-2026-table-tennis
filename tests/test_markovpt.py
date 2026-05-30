import pandas as pd

from scripts.produce_markovpt_oof import (
    observable_transition_samples,
    prediction_rows,
    visible_before_cuts,
)


def _strokes():
    return pd.DataFrame({
        "rally_uid": [1, 1, 1, 1],
        "strikeNumber": [1, 2, 3, 4],
        "sex": [1, 1, 1, 1],
        "match": [1, 1, 1, 1],
        "numberGame": [1, 1, 1, 1],
        "rally_id": [1, 1, 1, 1],
        "scoreSelf": [0, 0, 0, 0],
        "scoreOther": [0, 0, 0, 0],
        "serverGetPoint": [1, 1, 1, 1],
        "gamePlayerId": [10, 11, 10, 11],
        "gamePlayerOtherId": [11, 10, 11, 10],
        "strikeId": [0, 0, 0, 0],
        "handId": [0, 0, 0, 0],
        "strengthId": [0, 0, 0, 0],
        "spinId": [0, 0, 0, 0],
        "pointId": [1, 2, 3, 0],
        "actionId": [15, 1, 2, 3],
        "positionId": [0, 1, 2, 3],
    })


def test_observable_transition_samples_uses_all_nonfirst_strokes():
    out = observable_transition_samples(_strokes())
    assert out["target_strikeNumber"].tolist() == [2, 3, 4]
    assert out["y_actionId"].tolist() == [1, 2, 3]


def test_observable_transition_samples_excludes_hidden_cut_and_future():
    out = observable_transition_samples(_strokes(), {1: 3})
    assert out["target_strikeNumber"].tolist() == [2]
    assert out["y_actionId"].tolist() == [1]


def test_visible_before_cuts_matches_direct_truncation():
    all_transitions = observable_transition_samples(_strokes())
    visible = visible_before_cuts(all_transitions, {1: 3})
    assert visible["target_strikeNumber"].tolist() == [2]


def test_prediction_rows_uses_last_observable_stroke_before_cut():
    out = prediction_rows(_strokes(), {1: 3})
    assert out["target_strikeNumber"].tolist() == [3]
    assert out["last1_actionId"].tolist() == [1]
    assert out["next_gamePlayerId_inferred"].tolist() == [10]
