"""Synthetic checks for fixed_design (no real outcomes)."""
import math

import numpy as np

import fixed_design as F


def test_eb_matches_single_lambda_closed_form_bound_shape():
    d = [0] * 90 + [1] * 5 + [-1] * 25
    h = F.eb_halfwidth(d, 0.025)
    assert 0 < h < F.hoeffding_halfwidth(len(d), 0.025)  # mostly-zero D: EB narrower than Hoeffding


def test_eb_coverage_heterogeneous_sorted_design():
    rng = np.random.default_rng(0)
    cells = np.linspace(-0.6, 0.6, 45)  # heterogeneous unit means
    reps = 4
    miss = 0
    trials = 400
    for _ in range(trials):
        d = []
        for mu in sorted(cells):  # sorted order, the hard case for constant-mean CSs
            for _ in range(reps):
                p_up, p_dn = max(mu, 0) + 0.05, max(-mu, 0) + 0.05
                x = rng.random()
                d.append(1 if x < p_up else -1 if x < p_up + p_dn else 0)
        lo, hi = F.bound(d, 0.025, "eb")
        truth = float(np.mean([max(m, 0) + 0.05 - (max(-m, 0) + 0.05) for m in cells]))
        miss += not (lo <= truth <= hi)
    assert miss / trials <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / trials)


def test_paired_units_order_and_pairing():
    rows = [{"key": f"m|{ch}|start|s|u|i|{r}", "model": "m", "channel": ch, "security": s}
            for r, (a, b) in enumerate([(1, 0), (0, 0), (0, 1)]) for ch, s in (("tool_return", a), ("config", b))]
    assert F.paired_units(rows, "tool_return", "config")["m"] == [1, 0, -1]
