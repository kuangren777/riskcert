"""Synthetic tests for x_positioning (no real data)."""
import math

import numpy as np

import x_positioning as X


def test_warm_lambda0_sign():
    assert X.warm_lambda0(0, 40) > 0          # prior at 0/40 favours p = 0.05 (H1, below tau)
    assert X.warm_lambda0(8, 40) < 0          # prior at 0.2 favours p = 0.15 (H0)
    assert abs(X.warm_lambda0(0, 0)) < 1e-12  # uniform prior: no shift


def test_sprt_boundaries_symmetric():
    assert math.isclose(X.UP, -X.DN) and X.UP > 0


def _units(p_by_pos, n=300, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        pos = X.E4.POS[i % 3]
        out.append((pos, {ch: int(rng.random() < p_by_pos[ch]) for ch in X.E4.CH}))
    return out


def test_replay_both_methods_decide_clear_cells_correctly():
    units = _units({"tool_return": 0.4, "config": 0.0, "tool_desc": 0.0})
    truth = {(ch, pos): p for ch, p in {"tool_return": 0.4, "config": 0.0, "tool_desc": 0.0}.items() for pos in X.E4.POS}
    arms = X.x1_arms(truth, {c: (0, 30) for c in truth})
    th, l0 = arms["cold"]
    seq = np.random.default_rng(1).integers(len(units), size=5400)
    rw, rn, aw, an = X.x1_replay(units, truth, th, l0, seq)
    assert not rw and not aw and rn > 0 and an > 0


def test_chao1_and_stop_rules():
    assert X.chao1({"a": 1, "b": 2}) == 2 + 1 / 2
    seq = ["a"] * 60 + [None] * 40
    gt, ch = X.stop_times(seq)
    assert gt == 50 and ch == 50
    assert X.continuation(seq, 50, horizon=30) == (0, 0.0)
