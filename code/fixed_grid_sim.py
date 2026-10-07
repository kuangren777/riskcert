"""Fixed-grid vs i.i.d.-audit simulation for RQ2 (when do the certificates fail). Synthetic; no model runs.

Designs (theory/EB_HETEROGENEOUS.md §5, true design average delta_bar = 0 in all three): 45 units, r = 4 reps each,
N = 180 paired runs. A unit u has P(D=+1) = p+_u, P(D=-1) = p-_u, otherwise concordant (D = 0), D = X_a - X_b.
  A_mild     delta_u evenly spaced in [-0.2, 0.2], discordance pi_u = |delta_u| + 0.1
  B_extreme  15 units delta = +0.6, 30 units delta = -0.3, pi = |delta|
  D_adv      15 units delta = +0.3 (pi = 0.3), 30 units delta = -0.15 with pi = 0.95
Protocols:
  fixed-desc / fixed-asc   the grid in sorted order of delta_u (reps contiguous), the worst benchmark-file order
  fixed-perm               the grid in one uniformly random order per replay
  iid                      the i.i.d. audit protocol: each of the N rounds draws a unit uniformly with replacement
Methods (alpha = 0.05, one two-sided decision):
  R1-final   ContrastCS + RC-mix read once at N; error = interval excludes the design average 0 (Prop 5b: valid)
  R1-any     the same read after every round; error = excluded at some round (valid only under iid)
  R1'-final  SignCS read once at N; error = a sign is certified (delta_bar = 0)
  R1'-any    SignCS read after every round; error = a sign is certified at some round
  R1'mix-*   SignCS with the constant-bet mixture (RCMixCS) instead of the plug-in bet, final read / any time
             (round-3 review: at q = 1/2 its capital equals the comparison's, so the final read is valid)
  EB-final   empirical-Bernstein lambda-mixture at N (fixed_design.eb_halfwidth)

    python3 fixed_grid_sim.py [reps]   -> results/fixed_grid_sim.json, results/fixed_grid_sim.md
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fixed_design import eb_halfwidth
from riskcert import BettingCS, ContrastCS, RCMixCS, SignCS, sign_decision

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ALPHA, R, SEED = 0.05, 4, 2027
PROTOCOLS = ("fixed-desc", "fixed-asc", "fixed-perm", "iid")
METHODS = ("R1-final", "R1-any", "R1'-final", "R1'-any", "R1'mix-final", "R1'mix-any", "EB-final")


def design(name):
    if name == "A_mild":
        d = np.linspace(-0.2, 0.2, 45)
        pi = np.abs(d) + 0.1
    elif name == "B_extreme":
        d = np.r_[np.full(15, 0.6), np.full(30, -0.3)]
        pi = np.abs(d)
    elif name == "D_adv":
        d = np.r_[np.full(15, 0.3), np.full(30, -0.15)]
        pi = np.r_[np.full(15, 0.3), np.full(30, 0.95)]
    else:
        raise ValueError(name)
    return (pi + d) / 2, (pi - d) / 2  # p+, p-


def unit_order(p_plus, p_minus, protocol, rng):
    n_units, n = len(p_plus), len(p_plus) * R
    if protocol == "iid":
        return rng.integers(n_units, size=n)
    grid = np.repeat(np.arange(n_units), R)
    delta = (p_plus - p_minus)[grid]
    if protocol == "fixed-desc":
        return grid[np.argsort(-delta, kind="stable")]
    if protocol == "fixed-asc":
        return grid[np.argsort(delta, kind="stable")]
    return rng.permutation(grid)


def one_replay(p_plus, p_minus, protocol, rng):
    units = unit_order(p_plus, p_minus, protocol, rng)
    u = rng.random(len(units))
    d = np.where(u < p_plus[units], 1, np.where(u < p_plus[units] + p_minus[units], -1, 0))
    r1, sg, sm = ContrastCS(ALPHA), SignCS(ALPHA, cls=BettingCS), SignCS(ALPHA, cls=RCMixCS)
    target = 0.0  # design average and i.i.d. population mean are both 0 in every design
    r1_any = sg_any = sm_any = False
    for x in d:
        xa, xb = (1.0, 0.0) if x == 1 else (0.0, 1.0) if x == -1 else (0.0, 0.0)
        r1.update(xa, xb)
        sg.update(xa, xb)
        sm.update(xa, xb)
        sm_any |= sign_decision(*sm.interval()) is not None
        lo, hi = r1.interval()
        r1_any |= not (lo <= target <= hi)
        sg_any |= sign_decision(*sg.interval()) is not None
    lo, hi = r1.interval()
    m, h = float(np.mean(d)), eb_halfwidth(d.tolist(), ALPHA / 2)
    return {"R1-final": not (lo <= target <= hi), "R1-any": r1_any, "R1'-final": sign_decision(*sg.interval()) is not None,
            "R1'-any": sg_any,
            "R1'mix-final": sign_decision(*sm.interval()) is not None, "R1'mix-any": sm_any, "EB-final": not (m - h <= target <= m + h)}


def run_cell(args):
    name, protocol, reps, seed = args
    rng = np.random.default_rng(seed)
    pp, pm = design(name)
    errs = {k: 0 for k in METHODS}
    for _ in range(reps):
        for k, v in one_replay(pp, pm, protocol, rng).items():
            errs[k] += bool(v)
    return name, protocol, {k: v / reps for k, v in errs.items()}


def main(reps=1000):
    cells = [(n, p, reps, SEED + 10 * i + j) for i, n in enumerate(("A_mild", "B_extreme", "D_adv"))
             for j, p in enumerate(PROTOCOLS)]
    with ProcessPoolExecutor(min(12, len(cells))) as ex:
        out = list(ex.map(run_cell, cells))
    res = {"reps": reps, "alpha": ALPHA, "N": 45 * R, "seed": SEED, "rows": [
        {"design": n, "protocol": p, **e} for n, p, e in out]}
    md = [f"# Fixed grid vs i.i.d. audit: error rate at nominal {ALPHA} ({reps} replays per row, N = {45 * R})", "",
          "| design | protocol | " + " | ".join(METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
    for r in res["rows"]:
        md.append(f"| {r['design']} | {r['protocol']} | " + " | ".join(f"{r[k]:.3f}" for k in METHODS) + " |")
    json.dump(res, open(os.path.join(RES, "fixed_grid_sim.json"), "w"), indent=1)
    open(os.path.join(RES, "fixed_grid_sim.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
