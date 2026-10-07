"""Synthetic tests for fixed_grid_sim (fast, small replay counts)."""
import numpy as np

import fixed_grid_sim as F


def test_designs_have_zero_average():
    for n in ("A_mild", "B_extreme", "D_adv"):
        pp, pm = F.design(n)
        assert abs(np.mean(pp - pm)) < 1e-12 and (pp >= 0).all() and (pm >= 0).all() and (pp + pm <= 1).all()


def test_orders():
    pp, pm = F.design("B_extreme")
    rng = np.random.default_rng(0)
    desc = F.unit_order(pp, pm, "fixed-desc", rng)
    assert len(desc) == 180 and (pp - pm)[desc[0]] == 0.6 and (pp - pm)[desc[-1]] == -0.3
    assert sorted(F.unit_order(pp, pm, "fixed-perm", rng)) == sorted(desc)
    assert F.unit_order(pp, pm, "iid", rng).max() < 45


def test_sorted_extreme_breaks_sign_cs_not_r1_final():
    _, _, e = F.run_cell(("B_extreme", "fixed-desc", 20, 1))
    assert e["R1'-any"] >= 0.8 and e["R1-final"] <= 0.1 and e["EB-final"] <= 0.1
