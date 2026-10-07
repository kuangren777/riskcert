import numpy as np

import rc_replay as R
from riskcert import ContrastCS, paired_sprt_step, sign_decision, three_way, zone_decision


def _table(pa, pb, pc, n=200, seed=0, rho_unit=0.0):
    rng = np.random.default_rng(seed)
    t = {}
    for i in range(n):
        z = rng.random()
        shared = rng.random() < rho_unit
        t[str(i)] = {s: int((z if shared else rng.random()) < p)
                     for s, p in (("tool_return", pa), ("config", pb), ("tool_desc", pc))}
    return {"m": t}


def test_rule_primitives():
    assert sign_decision(0.1, 0.3) == ">" and sign_decision(-0.3, -0.1) == "<"
    assert sign_decision(-0.02, 0.03, eps=0.05) == "tie" and sign_decision(-0.1, 0.1, eps=0.05) is None
    assert zone_decision(0.0, 0.35, 0.1, 0.4) == "below" and zone_decision(0.15, 0.9, 0.1, 0.4) == "above"
    assert three_way(0.18, 0.22, 0.2, 0.05) == "near" and three_way(0.3, 0.5, 0.2, 0.05) == "above"
    assert paired_sprt_step(0.0, 1, 1, 0.2) == 0.0 and paired_sprt_step(0.0, 1, 0, 0.2) > 0


def test_contrast_cs_maps_back_and_covers():
    rng = np.random.default_rng(1)
    miss = 0
    for _ in range(100):
        c = ContrastCS(0.1)
        for _ in range(80):
            c.update(float(rng.random() < 0.5), float(rng.random() < 0.3))
            lo, hi = c.interval()
            if not lo <= 0.2 <= hi:
                miss += 1
                break
    assert -1 <= lo <= hi <= 1 and miss / 100 <= 0.1 + 3 * np.sqrt(0.09 / 100)


def test_pools_keep_only_complete_units():
    rows = [{"key": "m|tool_return|start|s|u|i|0", "model": "m", "channel": "tool_return", "security": True},
            {"key": "m|config|start|s|u|i|0", "model": "m", "channel": "config", "security": False},
            {"key": "m|tool_desc|start|s|u|i|0", "model": "m", "channel": "tool_desc", "security": False},
            {"key": "m|tool_return|end|s|u|i|0", "model": "m", "channel": "tool_return", "security": True}]
    t = R.pools_from_rows(rows)
    assert list(t["m"]) == ["start|s|u|i|0"]


def test_riskcert_policies_decide_clear_contrasts_correctly():
    t = _table(0.6, 0.1, 0.1)
    decs = R.decisions_for(t)
    tru = {d: R.truth(t, d) for d in decs}
    for pol in ("pairs", "open_strata"):
        cost, out = R.riskcert(t, decs, 0.05, np.random.default_rng(2), pol, eps=0.1, n_max=400)
        wrong, right = R.score(out, tru)
        assert wrong == 0 and right >= 2 and cost > 0


def test_open_strata_cheaper_when_contrasts_share_strata():
    t = _table(0.7, 0.1, 0.1)
    decs = [d for d in R.decisions_for(t) if d[1] == "tool_return"]  # both share tool_return
    c_pairs = np.mean([R.riskcert(t, decs, 0.05, np.random.default_rng(s), "pairs", n_max=400)[0] for s in range(5)])
    c_open = np.mean([R.riskcert(t, decs, 0.05, np.random.default_rng(s), "open_strata", n_max=400)[0] for s in range(5)])
    assert c_open < c_pairs


def test_sprt_and_fixed_baselines_run():
    t = _table(0.6, 0.1, 0.3)
    decs = R.decisions_for(t)
    c1, o1 = R.paired_sprt(t, decs, 0.05, np.random.default_rng(3), eta=0.2, n_max=400)
    c2, o2 = R.fixed_n(t, decs, 0.05, np.random.default_rng(3), n=100)
    assert c2 == 300 and set(o1) == set(o2) == set(decs)


def test_reversal_counting():
    decs = [("g1", "a", "b"), ("g2", "a", "b"), ("g3", "a", "b")]
    assert R.reversals({decs[0]: ">", decs[1]: "<", decs[2]: "<"}, decs) == 2
    assert R.reversals({decs[0]: ">", decs[1]: None, decs[2]: "tie"}, decs) == 0


def test_closure_costs_monotone_and_complete():
    t = _table(0.6, 0.1, 0.3)
    decs = R.decisions_for(t)
    cost, out = R.riskcert(t, decs, 0.05, np.random.default_rng(5), "open_strata", eps=0.1, n_max=300)
    ca = R.riskcert.closed_at
    assert set(ca) == set(decs) and max(ca.values()) == cost
    cost2, out2 = R.paired_sprt_rr(t, decs, 0.05, np.random.default_rng(5), 0.2, 300)
    assert set(R.paired_sprt_rr.closed_at) == set(decs) and max(R.paired_sprt_rr.closed_at.values()) == cost2


def test_reversals_with_truth_drop_wrong_signs():
    decs = [("g1", "a", "b"), ("g2", "a", "b")]
    tru = {decs[0]: 0.2, decs[1]: 0.1}  # both truly '>'
    assert R.reversals({decs[0]: ">", decs[1]: "<"}, decs) == 1
    assert R.reversals({decs[0]: ">", decs[1]: "<"}, decs, tru) == 0


def _three_channel_table(ps, n=120, seed=0):
    rng = np.random.default_rng(seed)
    return {f"m{j}": {str(i): {c: int(rng.random() < p) for c, p in zip(("tool_return", "config", "tool_desc"), pj)}
                      for i in range(n)} for j, pj in enumerate(ps)}


def test_peek_select_null_runs_and_flags_only_known_methods():
    t = _three_channel_table([(0.3, 0.3, 0.3), (0.2, 0.25, 0.2)])
    err, zone = R.peek_select(t, np.random.default_rng(0), "null", n_rounds=60)
    assert set(err) == {1, 5, 20}
    for L in err:
        assert set(err[L]) <= set(R.PEEK_METHODS)


def test_naive_peeking_inflates_fwer_under_null_more_than_r1prime():
    t = _three_channel_table([(0.4, 0.4, 0.4)] * 2, n=200)
    rng = np.random.default_rng(1)
    e_naive = e_r1p = 0
    for _ in range(40):
        err, _ = R.peek_select(t, rng, "null", looks=(20,), n_rounds=150, alpha=0.3)
        e_naive += err[20]["McNemar_peek"]
        e_r1p += err[20]["R1prime"]
    assert e_r1p <= e_naive
