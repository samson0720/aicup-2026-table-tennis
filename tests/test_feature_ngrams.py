import pandas as pd

from scripts.feature_ngrams import ngram_features


def test_bigram_hash_and_repeat_count():
    prefix = pd.DataFrame({
        "strikeNumber": [1, 2, 3, 4],
        "actionId": [5, 7, 7, 2],
        "spinId": [1, 1, 2, 3],
        "strikeId": [1, 1, 2, 2],
    })
    f = ngram_features(prefix)
    assert f["bigram_last_action"] == 702
    assert f["max_action_run"] == 2
    assert abs(f["strike_switch_rate"] - 1.0 / 3.0) < 1e-9
    assert f["action_transition_entropy"] > 0.0


def test_empty_like_prefix_defaults():
    prefix = pd.DataFrame(columns=["strikeNumber", "actionId", "spinId", "strikeId"])
    f = ngram_features(prefix)
    assert f["bigram_last_action"] == -1
    assert f["bigram_last_spin_action"] == -1
    assert f["max_action_run"] == 0
    assert f["strike_switch_rate"] == 0.0
