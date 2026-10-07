"""Synthetic tests for e4_replay (no real outcomes)."""
import numpy as np

import e4_replay as E


def _units(ps, n=120, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        pos = E.POS[i % 3]
        out.append((pos, {ch: int(rng.random() < ps[(ch, pos)]) for ch in E.CH}))
    return out


P = {(ch, pos): {"tool_return": 0.5, "config": 0.02, "tool_desc": 0.2}[ch] for ch in E.CH for pos in E.POS}


def test_thetas_shapes_and_sides():
    tc, tcon = E.truths(_units(P))
    for arm in E.ARMS:
        th_c, th_s = E.thetas(arm, tc, tcon, tc, tcon)
        assert set(th_c) == set(tc) and set(th_s) == set(E.CONTRASTS)
    th_o, _ = E.thetas("oracle", tc, tcon, tc, tcon)
    th_a, _ = E.thetas("adversary", tc, tcon, tc, tcon)
    c_hi = ("tool_return", "start")
    assert np.all(th_o[c_hi] == 0.8) and np.all(th_a[c_hi] == 0.2)


def test_run_arm_decides_and_is_correct_on_clear_pool():
    units = _units(P, n=240)
    tc, tcon = E.truths(units)
    seq = np.random.default_rng(1).integers(len(units), size=6000)
    for arm in ("cold", "oracle"):
        cost, wrong, decided = E.run_arm(units, seq, *E.thetas(arm, tc, tcon, tc, tcon), tc, tcon)
        assert wrong == 0 and decided >= 6 and cost > 0


def test_oracle_not_costlier_than_adversary_on_average():
    units = _units(P, n=240)
    tc, tcon = E.truths(units)
    rng = np.random.default_rng(2)
    co, ca = [], []
    for _ in range(8):
        seq = rng.integers(len(units), size=6000)
        co.append(E.run_arm(units, seq, *E.thetas("oracle", tc, tcon, tc, tcon), tc, tcon)[0])
        ca.append(E.run_arm(units, seq, *E.thetas("adversary", tc, tcon, tc, tcon), tc, tcon)[0])
    assert np.mean(co) <= np.mean(ca)
