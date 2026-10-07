"""Synthetic tests for m3b_analyze (no real outcomes)."""
import m3b_analyze as M


def _arm(c, f):
    return {"mean_cost": c, "median_cost": c, "fwer": f, "mean_decided": 12.0, "cost": [c]}


def test_e4_rules():
    t = lambda o, p: {"prev": "a", "new": "b", "units": 90, "prev_prior_right": 10,
                      "arms": {"cold": _arm(100, 0.01), "prev": _arm(p, 0.01), "oracle": _arm(o, 0.0), "adversary": _arm(120, 0.02)}}
    res, _ = M.e4([t(85, 95), t(80, 99), t(89, 120), t(70, 90)])
    assert res["E4_V"] and res["E4_C"] and res["prev_hits"] == 3
    res2, _ = M.e4([t(95, 95), t(80, 99), t(89, 99), t(70, 90)])
    assert not res2["E4_C"]


def test_e5_cell_consistency_and_comparators():
    seq = [("t", ("a",)), ("t", ("b",))] + [None] * 1300
    a = M.analyse_cell(seq)
    assert a["stop_n"] is not None and a["bound"] <= 0.05 and a["new_in_cont"] == 0 and a["consistent"]
    assert a["rule50_stop"] == 52 and a["rule50_r"] == 0.0 and a["good_turing"] == 2 / a["stop_n"]
