"""N-gram and streak features over a rally prefix."""
from __future__ import annotations

import math
from collections import Counter

import pandas as pd


def ngram_features(prefix: pd.DataFrame) -> dict[str, float | int]:
    p = prefix.sort_values("strikeNumber").reset_index(drop=True)
    action = p["actionId"].astype(int).tolist()
    spin = p["spinId"].astype(int).tolist()
    strike = p["strikeId"].astype(int).tolist()
    feat: dict[str, float | int] = {}

    feat["bigram_last_action"] = int(action[-2] * 100 + action[-1]) if len(action) >= 2 else -1
    feat["bigram_last_spin_action"] = int(spin[-1] * 100 + action[-1]) if action and spin else -1

    if action:
        best = run = 1
        for i in range(1, len(action)):
            if action[i] == action[i - 1]:
                run += 1
                best = max(best, run)
            else:
                run = 1
        feat["max_action_run"] = best
    else:
        feat["max_action_run"] = 0

    if len(strike) >= 2:
        switches = sum(1 for i in range(1, len(strike)) if strike[i] != strike[i - 1])
        feat["strike_switch_rate"] = switches / (len(strike) - 1)
    else:
        feat["strike_switch_rate"] = 0.0

    if len(action) >= 2:
        transitions = Counter(zip(action[:-1], action[1:]))
        total = sum(transitions.values())
        feat["action_transition_entropy"] = -sum(
            (count / total) * math.log(count / total + 1e-12)
            for count in transitions.values()
        )
    else:
        feat["action_transition_entropy"] = 0.0

    return feat
