"""E-B: contract-matched baselines and power under nonzero effects (plan.md E-B, preregistered 5d6cc121).

Contract: two-sided sign of delta for one paired contrast at level a = 0.05/4, readable after every round up to
N_MAX = 450 rounds, i.i.d. rounds. A round is discordant with probability pi; a discordant round favours stratum a
with probability q (delta = pi (2q - 1)).

Methods: sign certificate (ours, SignCS + plug-in BettingCS), comparison (ours, ContrastCS + RC-mix), alpha-spending
group-sequential McNemar (O'Brien-Fleming shape, L = 18 looks every 25 rounds, c calibrated by null simulation),
mixture SPRT on the discordant +-1 stream (normal mixture, tau^2 = 1), fixed-n McNemar at N_MAX.

    python3 eb_baselines.py [reps]   -> results/eb_baselines.json, results/eb_baselines.md
"""
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import binomtest

from riskcert import BettingCS, ContrastCS, SignCS, sign_decision

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
A, N_MAX, LOOK_EVERY, TAU2, SEED, CAL_SIMS = 0.05 / 4, 450, 25, 1.0, 2028, 20000
L = N_MAX // LOOK_EVERY
PIS = (0.05, 0.10, 0.20, 0.35)
QS = (0.5, 0.65, 0.8, 0.95)
METHODS = ("sign (ours)", "comparison (ours)", "alpha-spending OBF", "mSPRT", "fixed-n McNemar")


def rounds(pi, q, rng, n=N_MAX):
    """+1 = discordant favouring a, -1 = discordant favouring b, 0 = concordant."""
    u = rng.random(n)
    return np.where(u < pi * q, 1, np.where(u < pi, -1, 0))


def _z(up, dn):
    d = up + dn
    return 0.0 if d == 0 else (up - dn) / math.sqrt(d)


def calibrate_obf(pi, rng, sims=CAL_SIMS):
    """c such that P(max_k |z_k| / sqrt(L/k) >= c) = A under q = 1/2."""
    stats = np.empty(sims)
    for s in range(sims):
        y = rounds(pi, 0.5, rng)
        cu, cd = np.cumsum(y == 1), np.cumsum(y == -1)
        m = 0.0
        for k in range(1, L + 1):
            t = k * LOOK_EVERY - 1
            m = max(m, abs(_z(cu[t], cd[t])) / math.sqrt(L / k))
        stats[s] = m
    return float(np.quantile(stats, 1 - A))


def msprt_log_lr(s, n):
    """log normal-mixture likelihood ratio for the mean of a 1-sub-Gaussian +-1 stream after n discordant steps."""
    return 0.5 * math.log(1.0 / (1.0 + n * TAU2)) + TAU2 * s * s / (2.0 * (1.0 + n * TAU2))


def run_one(y, c_obf):
    """Decision (sign or None) and decision round (1-based) per method for one replay."""
    out = {m: (None, None) for m in METHODS}
    sg, cp = SignCS(A, cls=BettingCS), ContrastCS(A)
    up = dn = 0
    thr = math.log(1 / A)
    for t, x in enumerate(y, 1):
        xa, xb = (1.0, 0.0) if x == 1 else (0.0, 1.0) if x == -1 else (0.0, 0.0)
        up += x == 1
        dn += x == -1
        if out["sign (ours)"][0] is None:
            sg.update(xa, xb)
            d = sign_decision(*sg.interval())
            if d:
                out["sign (ours)"] = (d, t)
        if out["comparison (ours)"][0] is None:
            cp.update(xa, xb)
            d = sign_decision(*cp.interval())
            if d:
                out["comparison (ours)"] = (d, t)
        if out["mSPRT"][0] is None and x != 0 and msprt_log_lr(up - dn, up + dn) >= thr:
            out["mSPRT"] = (">" if up > dn else "<", t)
        if out["alpha-spending OBF"][0] is None and t % LOOK_EVERY == 0:
            k = t // LOOK_EVERY
            z = _z(up, dn)
            if abs(z) >= c_obf * math.sqrt(L / k):
                out["alpha-spending OBF"] = (">" if z > 0 else "<", t)
    if up + dn and binomtest(int(up), int(up + dn), 0.5).pvalue <= A:
        out["fixed-n McNemar"] = (">" if up > dn else "<", N_MAX)
    return out


def run_setting(args):
    pi, q, reps, seed = args
    rng = np.random.default_rng(seed)
    c = calibrate_obf(pi, np.random.default_rng(seed + 1))
    truth = None if q == 0.5 else ">"
    acc = {m: {"err": 0, "power": 0, "abstain": 0, "lat": []} for m in METHODS}
    for _ in range(reps):
        res = run_one(rounds(pi, q, rng), c)
        for m, (d, t) in res.items():
            if d is None:
                acc[m]["abstain"] += 1
            elif d == truth:
                acc[m]["power"] += 1
                acc[m]["lat"].append(t)
            else:
                acc[m]["err"] += 1
    row = {"pi": pi, "q": q, "delta": pi * (2 * q - 1), "c_obf": c, "reps": reps, "methods": {}}
    for m, v in acc.items():
        lat = v["lat"]
        cost = 2 * (sum(lat) + N_MAX * (reps - len(lat))) / reps
        row["methods"][m] = {"error": v["err"] / reps, "power": v["power"] / reps, "abstain": v["abstain"] / reps,
                             "median_latency": float(np.median(lat)) if lat else None,
                             "mean_latency": float(np.mean(lat)) if lat else None, "mean_episode_cost": cost}
    return row


def main(reps=1000):
    cells = [(pi, q, reps, SEED + 100 * i + j) for i, pi in enumerate(PIS) for j, q in enumerate(QS)]
    with ProcessPoolExecutor(min(16, len(cells))) as ex:
        rows = list(ex.map(run_setting, cells))
    se = math.sqrt(A * (1 - A) / reps)
    valid = {m: all(r["methods"][m]["error"] <= A + 3 * se for r in rows) for m in METHODS}
    nonnull = [r for r in rows if r["q"] != 0.5]
    wins = sum(r["methods"]["sign (ours)"]["power"] >= r["methods"]["alpha-spending OBF"]["power"] for r in nonnull)
    res = {"a": A, "n_max": N_MAX, "reps": reps, "seed": SEED, "rows": rows, "B_valid": valid,
           "B_power_sign_ge_obf": f"{wins}/{len(nonnull)}",
           "B_power_settings_lost": [(r["pi"], r["q"]) for r in nonnull
                                     if r["methods"]["sign (ours)"]["power"] < r["methods"]["alpha-spending OBF"]["power"]]}
    md = [f"# E-B: contract-matched baselines (a = {A:.4f}, N_max = {N_MAX}, {reps} replays per setting)", "",
          "| pi | q | delta | method | error | power | abstain | median latency | mean episode cost |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for m in METHODS:
            v = r["methods"][m]
            md.append(f"| {r['pi']} | {r['q']} | {r['delta']:.3f} | {m} | {v['error']:.3f} | {v['power']:.3f} | {v['abstain']:.3f} | "
                      f"{v['median_latency'] if v['median_latency'] is not None else '-'} | {v['mean_episode_cost']:.0f} |")
    md += ["", f"B-valid (error <= a + 3 s.e. everywhere): {valid}",
           f"B-power: sign certificate >= alpha-spending power in {res['B_power_sign_ge_obf']} non-null settings; lost: {res['B_power_settings_lost']}"]
    json.dump(res, open(os.path.join(RES, "eb_baselines.json"), "w"), indent=1)
    open(os.path.join(RES, "eb_baselines.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md[-2:]))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
