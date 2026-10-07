"""Synthetic tests for c3_run / c3_analyze (no real outcomes, no network)."""
import numpy as np

import c3_analyze as A
import c3_run as C


def _seq(p_tr_only, p_cfg_only, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        u = rng.random()
        out.append((1, 0) if u < p_tr_only else (0, 1) if u < p_tr_only + p_cfg_only else (int(rng.random() < 0.1),) * 2)
    return out


def test_frozen_units_and_draws():
    u = C.units()
    assert len(u) == 1023 and len(set(u)) == 1023
    d0, d1 = C.draws(0), C.draws(1)
    assert len(d0) == C.N_MAX and d0 == C.draws(0) and d0 != d1 and max(d0) < 1023


def test_qwen_cap_is_8():
    import rc_run
    assert rc_run.Gate("qwen3-8b-local").burst == rc_run.MAX_CONC == 8


def test_stopper_certifies_clear_effect_both_signs():
    r = A.analyse(_seq(0.15, 0.01, 450, 0), ">")
    assert r["decision"] == ">" and r["rounds"] < 450 and r["p1"]
    r = A.analyse(_seq(0.01, 0.15, 450, 1), ">")
    assert r["decision"] == "<" and r["p1_opposite"] and not r["p1"]


def test_null_effect_rarely_certifies():
    wrong = sum(A.analyse(_seq(0.05, 0.05, 450, s), ">")["decision"] is not None for s in range(100))
    assert wrong <= 5  # a = 0.0125 two-sided per contrast


def test_stop_at_nmax_and_eb_contains_mean():
    r = A.analyse(_seq(0.0, 0.0, 600, 3), ">")
    assert r["rounds"] == C.N_MAX and r["decision"] is None and r["p1"]
    assert r["eb_ci"][0] <= r["delta"] <= r["eb_ci"][1]


def test_paired_sequence_stops_at_gap_and_skips_errors():
    rows = [{"round": 0, "channel": "tool_return", "security": True}, {"round": 0, "channel": "config", "security": False},
            {"round": 1, "channel": "tool_return", "err": "x"}, {"round": 1, "channel": "config", "security": True},
            {"round": 2, "channel": "tool_return", "security": True}, {"round": 2, "channel": "config", "security": True}]
    assert A.paired_sequence(rows) == [(1, 0)]


def test_p2():
    res = {"qwen3-8b-local": {"decision": ">"}, "glm-5.3": {"decision": "<"}, "deepseek-v4.1-flash": {"decision": None}}
    assert A.p2(res) == (True, ["glm-5.3"])
    res["qwen3-8b-local"]["decision"] = None
    assert not A.p2(res)[0]


def test_r1_posthoc_present_and_agrees_on_clear_effect():
    r = A.analyse(_seq(0.3, 0.0, 450, 4), ">")
    assert "r1_posthoc_sign" in r and r["r1_posthoc_sign"] in (">", None)
    assert r["r1_posthoc_interval"][0] <= r["delta"] <= r["r1_posthoc_interval"][1]
