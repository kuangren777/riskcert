import numpy as np
import pytest

from riskcert import (GRID, BettingCS, RCMixCS, SignCS, fixed_n, prior_theta, riskcert_threshold,
                      threshold_decision, wald_sprt)


def _anytime_miss(p, alpha, theta=0.5, reps=200, steps=150, seed=0, cls=BettingCS):
    rng = np.random.default_rng(seed)
    miss = 0
    for _ in range(reps):
        cs = cls(alpha, theta=theta)
        for _ in range(steps):
            cs.update(float(rng.random() < p))
            lo, hi = cs.interval()
            if not lo <= p <= hi:
                miss += 1
                break
    return miss / reps


@pytest.mark.parametrize("p", [0.02, 0.15, 0.5, 0.85])
def test_anytime_coverage(p):
    # time-uniform miscoverage must stay <= alpha (sampling slack ~ 3 sd at reps=200)
    assert _anytime_miss(p, alpha=0.1) <= 0.1 + 3 * np.sqrt(0.1 * 0.9 / 200)


def test_wrong_prior_keeps_coverage():
    # prior says p ~ 0.9 while truth is 0.1: costs power, never validity
    th = prior_theta(0.9, strength=0.95)
    assert _anytime_miss(0.1, alpha=0.1, theta=th, seed=1) <= 0.1 + 3 * np.sqrt(0.09 / 200)


def test_interval_shrinks_and_contains_mean():
    rng = np.random.default_rng(2)
    cs = BettingCS(0.05)
    widths = []
    for t in range(400):
        cs.update(float(rng.random() < 0.3))
        if t in (20, 100, 399):
            lo, hi = cs.interval()
            widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2]
    assert lo <= 0.3 <= hi


def test_threshold_decision():
    assert threshold_decision(0.3, 0.5, 0.2) == "above"
    assert threshold_decision(0.01, 0.1, 0.2) == "below"
    assert threshold_decision(0.1, 0.3, 0.2) is None


def test_fixed_n_reproduces_n13():
    assert fixed_n(0.05, 0.30, 0.05, 0.10) == (16, 3)


def test_sprt_decides_clear_cells():
    rng = np.random.default_rng(3)
    hi_pool, lo_pool = np.ones(20), np.zeros(20)
    assert wald_sprt(hi_pool, 0.05, 0.3, 0.05, 0.1, 100, rng)[1] == "above"
    assert wald_sprt(lo_pool, 0.05, 0.3, 0.05, 0.1, 100, rng)[1] == "below"


def test_riskcert_threshold_on_pools():
    rng = np.random.default_rng(4)
    pools = [np.array([1] * 18 + [0] * 2), np.zeros(20), np.array([1] * 4 + [0] * 16)]
    n, d = riskcert_threshold(pools, tau=0.2, alpha=0.05, n_max=60, rng=rng)
    assert d[0] == "above" and d[1] == "below"
    assert n[2] == 60 and d[2] is None  # p = tau: never certified


@pytest.mark.parametrize("p", [0.0123, 0.3337])  # off-grid means
def test_rcmix_anytime_coverage_off_grid(p):
    assert _anytime_miss(p, alpha=0.1, cls=RCMixCS, steps=100) <= 0.1 + 3 * np.sqrt(0.1 * 0.9 / 200)


def test_grid_closure_contains_every_continuous_point():
    # dense check: any m whose exact capital is < 1/alpha must lie inside the returned interval
    rng = np.random.default_rng(7)
    for cls in (BettingCS, RCMixCS):
        cs = cls(0.05, grid=np.linspace(0, 1, 41))
        fine = cls(0.05, grid=np.linspace(0, 1, 4001))
        for _ in range(60):
            x = float(rng.random() < 0.3)
            if cls is BettingCS:  # same bet sequence on both grids: the plug-in lambda ignores m
                lam = cs._lam()
                fine._lam = lambda lam=lam: lam
            cs.update(x)
            fine.update(x)
            lo, hi = cs.interval()
            ins = fine.grid[fine.inside()]
            assert ins.min() >= lo - 1e-12 and ins.max() <= hi + 1e-12


def test_capital_monotone_in_m():
    rng = np.random.default_rng(8)
    for cs in (BettingCS(0.05), RCMixCS(0.05)):
        for _ in range(50):
            cs.update(float(rng.random() < 0.4))
        lkp, lkm = cs._logk()
        assert np.all(np.diff(lkp) <= 1e-9) and np.all(np.diff(lkm) >= -1e-9)


def test_rcmix_decides_clear_cells():
    rng = np.random.default_rng(9)
    pools = [np.array([1] * 18 + [0] * 2), np.zeros(20)]
    n, d = riskcert_threshold(pools, tau=0.2, alpha=0.05, n_max=200, rng=rng, cs_cls=RCMixCS)
    assert d == ["above", "below"] and max(n) < 200


def test_signcs_ignores_concordant_and_covers_sign():
    rng = np.random.default_rng(11)
    wrong = 0
    for _ in range(100):  # delta = 0.3 - 0.2 > 0, independent strata
        c = SignCS(0.1)
        for _ in range(300):
            c.update(float(rng.random() < 0.3), float(rng.random() < 0.2))
            lo, hi = c.interval()
            if hi < 0:  # certified the wrong sign at some time
                wrong += 1
                break
    assert wrong / 100 <= 0.1
    c = SignCS(0.05)
    for _ in range(50):
        c.update(1.0, 1.0)
    assert c.cs.n == 0 and c.n == 50
