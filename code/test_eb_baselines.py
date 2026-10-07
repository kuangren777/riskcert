"""Synthetic tests for eb_baselines (small, fast)."""
import math

import numpy as np

import eb_baselines as E


def test_rounds_rates():
    y = E.rounds(0.2, 0.8, np.random.default_rng(0), n=200000)
    assert abs(np.mean(y != 0) - 0.2) < 0.01 and abs(np.mean(y == 1) / np.mean(y != 0) - 0.8) < 0.01


def test_msprt_lr_zero_at_start_and_grows():
    assert E.msprt_log_lr(0, 0) == 0.0
    assert E.msprt_log_lr(40, 40) > math.log(1 / E.A) > E.msprt_log_lr(2, 40)


def test_strong_effect_all_sequential_methods_decide_correctly():
    rng = np.random.default_rng(1)
    c = E.calibrate_obf(0.35, np.random.default_rng(2), sims=2000)
    out = E.run_one(E.rounds(0.35, 0.95, rng), c)
    for m in ("sign (ours)", "alpha-spending OBF", "mSPRT", "fixed-n McNemar"):
        assert out[m][0] == ">", m


def test_null_rarely_errs():
    row = E.run_setting((0.2, 0.5, 60, 3))
    for m, v in row["methods"].items():
        assert v["error"] <= 0.1, m
