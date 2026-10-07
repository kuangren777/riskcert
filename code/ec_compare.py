"""E-C: replay comparison with Rank CS and BB-EDGE (plan.md E-C, preregistered 58827ec2).

K = 27 within-model channel contrasts of session B, family-wise alpha = 0.05, read after every round, i.i.d. unit
draws per model. Settings 'real' (pool truth) and 'null' (per-draw channel permutation, delta = 0).
Algorithm specs: review/rank_cs_2609.32211.md (Rank CS, Sec 3 eq 2-3, e-Bonferroni) and review/bb_edge_2609.32248.md
(BB-EDGE, Sec 4 eq 3-8, direct e-Holm), adapted as stated in plan.md E-C.

    python3 ec_compare.py [reps] [rounds]   -> results/ec_compare.json, results/ec_compare.md
"""
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import rc_replay
import sessb_analyze as S
from riskcert import BettingCS, ContrastCS, SignCS, sign_decision

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ALPHA, SEED = 0.05, 2029
RANK_GRID = np.array([0.03, 0.06, 0.12, 0.25, 0.5])
BB_GRID = np.linspace(0.0, 0.95, 41)
BB_PSI = -np.log1p(-BB_GRID) - BB_GRID
CHECK = (100, 250, 500, 1000)
METHODS = ("comparison (ours)", "sign (ours)", "Rank CS", "BB-EDGE")


class RankCS:
    """Both directions of one contrast; wealth = uniform grid average of prod(1 + lam * (+-Z)) (eq 2, b_t = 0)."""

    def __init__(self):
        self.lp = np.zeros_like(RANK_GRID)
        self.lm = np.zeros_like(RANK_GRID)

    def update(self, z):
        self.lp += np.log1p(RANK_GRID * z)
        self.lm += np.log1p(-RANK_GRID * z)

    def e(self):
        return float(np.mean(np.exp(self.lp))), float(np.mean(np.exp(self.lm)))


class BBEdge:
    """Both directions of one contrast, one block, one draw per replicate (eq 6-7)."""

    def __init__(self):
        self.lp = np.zeros_like(BB_GRID)
        self.lm = np.zeros_like(BB_GRID)
        self.n, self.s = 0, 0.0

    def update(self, x):
        yhat = (0.5 + self.s) / (self.n + 1)  # running mean initialised at mu_0 = 1/2, predictable
        self.lp += BB_GRID * (x - 0.5) - BB_PSI * (x - yhat) ** 2
        self.lm += BB_GRID * ((1 - x) - 0.5) - BB_PSI * ((1 - x) - (1 - yhat)) ** 2
        self.n += 1
        self.s += x

    def e(self):
        return float(np.mean(np.exp(self.lp))), float(np.mean(np.exp(self.lm)))


def e_holm(evals, alpha=ALPHA):
    """Direct e-Holm (eq 8): indices e with E_e >= c = 1/alpha + sum_{E_j < 1/alpha} (1/alpha - E_j)."""
    inv = 1 / alpha
    c = inv + sum(inv - v for v in evals if v < inv)
    return [i for i, v in enumerate(evals) if v >= c]


def one_replay(args):
    table, decs, setting, n_rounds, seed = args
    rng = np.random.default_rng(seed)
    k = len(decs)
    a = ALPHA / k
    units = {m: list(us.values()) for m, us in table.items()}
    models = sorted(units)
    cmp_ = {d: ContrastCS(a) for d in decs}
    sgn = {d: SignCS(a, cls=BettingCS) for d in decs}
    rk = {d: RankCS() for d in decs}
    bb = {d: BBEdge() for d in decs}
    dec = {m: {} for m in METHODS}  # method -> {d: (sign, round)}
    snap = {m: {} for m in METHODS}
    for t in range(1, n_rounds + 1):
        drawn = {}
        for m in models:
            u = dict(units[m][rng.integers(len(units[m]))])
            if setting == "null":
                chs = sorted(u)
                vals = [u[c] for c in chs]
                rng.shuffle(vals)
                u = dict(zip(chs, vals))
            drawn[m] = u
        rank_e, bb_e = [], []
        for d in decs:
            xa, xb = float(drawn[d[0]][d[1]]), float(drawn[d[0]][d[2]])
            if d not in dec["comparison (ours)"]:
                cmp_[d].update(xa, xb)
                s_ = sign_decision(*cmp_[d].interval())
                if s_:
                    dec["comparison (ours)"][d] = (s_, t)
            if d not in dec["sign (ours)"]:
                sgn[d].update(xa, xb)
                s_ = sign_decision(*sgn[d].interval())
                if s_:
                    dec["sign (ours)"][d] = (s_, t)
            rk[d].update(xa - xb)
            bb[d].update((xa - xb + 1) / 2)
            rank_e += list(rk[d].e())
            bb_e += list(bb[d].e())
        thr = 2 * k / ALPHA
        for i, d in enumerate(decs):
            if d not in dec["Rank CS"]:
                if rank_e[2 * i] >= thr:
                    dec["Rank CS"][d] = (">", t)
                elif rank_e[2 * i + 1] >= thr:
                    dec["Rank CS"][d] = ("<", t)
        for j in e_holm(bb_e):
            d = decs[j // 2]
            if d not in dec["BB-EDGE"]:
                dec["BB-EDGE"][d] = (">" if j % 2 == 0 else "<", t)
        if t in CHECK:
            for m in METHODS:
                snap[m][t] = dict(dec[m])
    return dec, snap


def truth_sign(table, d, setting):
    if setting == "null":
        return None
    v = rc_replay.truth(table, d)
    return ">" if v > 0 else "<" if v < 0 else None


def main(reps=200, n_rounds=1000):
    table = rc_replay.pools_from_rows(S.load_ad("B"))
    decs = rc_replay.decisions_for(table)
    out = {"alpha": ALPHA, "K": len(decs), "reps": reps, "rounds": n_rounds, "seed": SEED, "settings": {}}
    md = [f"# E-C: replay comparison (K = {len(decs)}, alpha = {ALPHA}, {reps} replays x {n_rounds} rounds)", ""]
    for si, setting in enumerate(("real", "null")):
        jobs = [(table, decs, setting, n_rounds, SEED + 10000 * si + r) for r in range(reps)]
        with ProcessPoolExecutor(16) as ex:
            res = list(ex.map(one_replay, jobs))
        tru = {d: truth_sign(table, d, setting) for d in decs}
        row = {}
        for m in METHODS:
            fw = sum(any(s != tru[d] for d, (s, _) in dec[m].items()) for dec, _ in res) / reps
            corr = {t: float(np.mean([sum(s == tru[d] and tru[d] is not None for d, (s, _) in sn[m].get(t, {}).items())
                                      for _, sn in res])) for t in CHECK if t <= n_rounds}
            lat = [r_ for dec, _ in res for d, (s, r_) in dec[m].items() if s == tru[d]]
            row[m] = {"fwer": fw, "correct_at": corr, "mean_round_correct": float(np.mean(lat)) if lat else None}
        out["settings"][setting] = row
        md += [f"## {setting}", "", "| method | FWER | " + " | ".join(f"correct @{t}" for t in CHECK if t <= n_rounds) + " | mean round |",
               "|---|---|" + "---|" * (len([t for t in CHECK if t <= n_rounds]) + 1)]
        for m, v in row.items():
            md.append(f"| {m} | {v['fwer']:.3f} | " + " | ".join(f"{v['correct_at'][t]:.2f}" for t in v["correct_at"])
                      + f" | {v['mean_round_correct'] if v['mean_round_correct'] is None else round(v['mean_round_correct'], 1)} |")
        md.append("")
    se = math.sqrt(ALPHA * (1 - ALPHA) / reps)
    out["C_valid"] = {m: all(out["settings"][s][m]["fwer"] <= ALPHA + 3 * se for s in out["settings"]) for m in METHODS}
    r500 = {m: out["settings"]["real"][m]["correct_at"].get(500) for m in METHODS}
    out["C_power"] = {"sign_at_500": r500["sign (ours)"], "rank_at_500": r500["Rank CS"], "bb_at_500": r500["BB-EDGE"],
                      "pass": r500["sign (ours)"] >= r500["Rank CS"] and r500["sign (ours)"] >= r500["BB-EDGE"]}
    md += [f"C-valid: {out['C_valid']}", f"C-power: {out['C_power']}"]
    json.dump(out, open(os.path.join(RES, "ec_compare.json"), "w"), indent=1)
    open(os.path.join(RES, "ec_compare.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    a = sys.argv
    main(int(a[1]) if len(a) > 1 else 200, int(a[2]) if len(a) > 2 else 1000)
