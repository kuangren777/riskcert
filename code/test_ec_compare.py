"""Synthetic tests for ec_compare (no real data)."""
import numpy as np

import ec_compare as E


def _table(p_up=0.4, n=60, seed=0):
    rng = np.random.default_rng(seed)
    us = {}
    for i in range(n):
        tr = int(rng.random() < p_up)
        us[f"u{i}"] = {"tool_return": tr, "config": 0, "tool_desc": int(rng.random() < 0.05)}
    return {"m1": us}


def test_e_holm_basic():
    assert E.e_holm([100.0, 1.0, 1.0], alpha=0.05) == [0]
    assert E.e_holm([10.0, 10.0], alpha=0.05) == []


def test_bbedge_and_rank_grow_on_signal_and_stay_small_on_null():
    r, b = E.RankCS(), E.BBEdge()
    for _ in range(300):
        r.update(1.0)
        b.update(1.0)
    assert r.e()[0] > 1e6 and b.e()[0] > 1e6 and r.e()[1] < 1 and b.e()[1] < 1
    rng = np.random.default_rng(1)
    r, b = E.RankCS(), E.BBEdge()
    for z in rng.choice([-1.0, 0.0, 1.0], size=300, p=[0.1, 0.8, 0.1]):
        r.update(z)
        b.update((z + 1) / 2)
    assert max(r.e()) < 1 / 0.05 * 3 and max(b.e()) < 1 / 0.05 * 3


def test_replay_runs_and_certifies_clear_contrast():
    t = _table()
    decs = [("m1", "tool_return", "config")]
    dec, snap = E.one_replay((t, decs, "real", 250, 3))
    for m in E.METHODS:
        assert dec[m].get(decs[0], (">",))[0] == ">"
    assert set(snap[E.METHODS[0]]) == {100, 250}
