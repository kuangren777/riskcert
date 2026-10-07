"""Session-B analyses, preregistered in plan.md M3a (681cfb5d, amendments 52206bd5 / 91e57749 / 009f2c42).
Run ONCE, after every session-B model (incl. Qwen2.5-7B and Qwen3-32B on local_b) is complete.

  C2   R1' (frozen SignCS) vs R1 (ContrastCS, RC-mix) on session-B AgentDojo rows only; pass = R1' >= R1 + 4.
       Secondary: (a) sign agreement with session-A point estimates, (b) budget curve on session-B pools.
  E3'  peeking + post hoc selection FWER on session-B pools (rc_replay.peek_experiment), null and real settings.
  RQ2  coverage: (i) within session A, time-uniform, by resampling A against the full-A mean (primary contract);
       (ii) cross-session, final session-A interval vs session-B mean, with a parametric-bootstrap noise baseline.

    python3 sessb_analyze.py c2 | e3p | rq2 [--reps N]
Outputs results/{c2,e3p,rq2}.json + .md. Null-agent rows are excluded from every analysis (they are oracle controls).
"""
import collections
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import norm

import rc_replay
from riskcert import BettingCS, RCMixCS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES = os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "results")
NULL = "null-agent"
SESSION_A_AD = ("m0.jsonl", "pilot.jsonl", "pilot_llama31-8b-local.jsonl", "e1_qwen3.jsonl", "e1_controls.jsonl",
                "e1_qwen25.jsonl", "e1_qwen32.jsonl")


def _load(paths):
    rows, seen = [], set()
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            if r.get("err") is not None or r.get("security") is None or r["model"] == NULL or r["key"] in seen:
                continue
            if r["model"] == "qwen3-8b-local" and not r.get("extra_body"):
                continue  # M0 Qwen3 rows predate enable_thinking=false (plan.md)
            seen.add(r["key"])
            rows.append(r)
    return rows


def load_ad(session, data=DATA):
    if session == "A":
        return _load([os.path.join(data, f) for f in SESSION_A_AD])
    return _load(sorted(glob.glob(os.path.join(data, "e2b_*.jsonl"))))


def load_ia(session, data=DATA):
    pat = "ia_e1_*.jsonl" if session == "A" else "ia_e2b_*.jsonl"
    return _load(sorted(glob.glob(os.path.join(data, pat))))


def _write(name, res, md):
    json.dump(res, open(os.path.join(RES, f"{name}.json"), "w"), indent=1, default=str)
    open(os.path.join(RES, f"{name}.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


# ------------------------------------------------------------------------------------------------ C2

def c2(rows_b, rows_a, budget_reps=50):
    tb = rc_replay.pools_from_rows(rows_b)
    ta = rc_replay.pools_from_rows(rows_a)
    cert = rc_replay.certify(tb)
    n_r1 = sum(r["r1"] in (">", "<") for r in cert["rows"])
    n_r1p = sum(r["r1p"] in (">", "<") for r in cert["rows"])
    agree, disagree = 0, []
    for r in cert["rows"]:
        if r["r1p"] not in (">", "<"):
            continue
        d = (r["model"], r["a"], r["b"])
        if r["model"] not in ta:
            continue
        da = rc_replay.truth(ta, d)
        if da != 0 and (da > 0) == (r["r1p"] == ">"):
            agree += 1
        else:
            disagree.append({"contrast": "|".join(d), "sign_B": r["r1p"], "delta_A": da})
    res = {"K": cert["K"], "alpha": cert["alpha"], "certified_r1": n_r1, "certified_r1prime": n_r1p,
           "pass": n_r1p >= n_r1 + 4, "agree_with_A": agree, "disagree_with_A": disagree, "rows": cert["rows"],
           "reversals_r1p": cert["reversals_r1p"], "reversals_r1": cert["reversals_r1"],
           "budget_curve_B": rc_replay.budget_curve(tb, reps=budget_reps) if budget_reps else None}
    md = ["# C2: R1′ vs R1 on session-B AgentDojo (preregistered)", "",
          f"K = {res['K']} contrasts, α = {res['alpha']}", "",
          f"certified R1′ = {n_r1p}, certified R1 = {n_r1}, pass rule R1′ ≥ R1 + 4: **{'PASS' if res['pass'] else 'FAIL'}**", "",
          f"R1′ signs agreeing with session-A point estimate: {agree}/{n_r1p}; disagreements: {len(disagree)}", "",
          "| model | a − b | n | discordant | δ | R1 | R1′ |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['model']} | {r['a']} − {r['b']} | {r['n']} | {r['discordant']} | {r['delta']:+.3f} | {r['r1'] or '?'} | {r['r1p'] or '?'} |"
           for r in cert["rows"]]
    return res, md


# ------------------------------------------------------------------------------------------------ E3'

def e3p(rows_b, reps=400, n_rounds=500):
    tb = rc_replay.pools_from_rows(rows_b)
    res = rc_replay.peek_experiment(tb, reps=reps, n_rounds=n_rounds)
    md = ["# E3′: FWER under peeking and post hoc selection (preregistered, session-B pools)", "",
          f"K = {res['K']}, replays = {reps}, rounds = {n_rounds}; pass = R1 and R1′ FWER ≤ 0.083 in every setting and L", ""]
    ok = True
    for setting, byL in res["results"].items():
        md += [f"## {setting}", "", "| method | " + " | ".join(f"L={L}" for L in byL) + " |", "|---" * (len(byL) + 1) + "|"]
        for mth in rc_replay.PEEK_METHODS:
            cells = []
            for L, v in byL.items():
                f = v[mth]["fwer"]
                z = v[mth]["sprt_in_zone_errors"]
                cells.append(f"{f:.3f}" + (f" (in-zone {z:.3f})" if mth.startswith("SPRT") else ""))
                if mth in ("R1", "R1prime") and f > 0.083:
                    ok = False
            md.append(f"| {mth} | " + " | ".join(cells) + " |")
        md.append("")
    res["pass"] = ok
    md.append(f"**Verdict (preregistered): {'PASS' if ok else 'FAIL'}**")
    return res, md


# ------------------------------------------------------------------------------------------------ RQ2

def cells_ad(rows):
    c = collections.defaultdict(list)
    for r in rows:
        c[f"{r['model']}|{r['channel']}|{r['pos']}"].append(int(bool(r["security"])))
    return c


def cells_ia(rows, metric="asr_valid"):
    c = collections.defaultdict(list)
    for r in rows:
        if metric == "asr_valid" and r["eval"] == "invalid":
            continue
        c[f"{r['model']}|{r['attack']}|{r['setting']}"].append(int(r["eval"] == "succ"))
    return c


def _wilson(k, n, z):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def _cp(k, n, a):
    lo = 0.0 if k == 0 else beta_dist.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def within_session(cells, alpha=0.05, reps=200, seed=0):
    """Primary contract: resample each cell's session-A pool i.i.d. for n_cell steps, truth = pool mean. Time-uniform
    miss = the interval excludes the truth at any step. Reports marginal miss per method and simultaneous miss
    (any cell missed in a replay). Wilson is evaluated at every step (naive peeking); Clopper-Pearson only at the
    final n (its valid use)."""
    keys = sorted(cells)
    k = len(keys)
    a = alpha / k
    z = norm.isf(a / 2)
    rng = np.random.default_rng(seed)
    methods = ("RC-mix", "plug-in", "Wilson-peek", "CP-final")
    marg = {m: 0 for m in methods}
    simul = {m: 0 for m in methods}
    widths = {m: [] for m in methods}
    for _ in range(reps):
        any_miss = {m: False for m in methods}
        for key in keys:
            pool = np.array(cells[key], float)
            truth, n = pool.mean(), len(pool)
            xs = pool[rng.integers(n, size=n)]
            miss = {m: False for m in methods}
            cs = {"RC-mix": RCMixCS(a), "plug-in": BettingCS(a)}
            s = 0
            for t, x in enumerate(xs, 1):
                s += x
                for m, c in cs.items():
                    c.update(float(x))
                    lo, hi = c.interval()
                    miss[m] |= not lo - 1e-12 <= truth <= hi + 1e-12
                lo, hi = _wilson(s, t, z)
                miss["Wilson-peek"] |= not lo - 1e-12 <= truth <= hi + 1e-12
            lo, hi = _cp(int(s), n, a)
            miss["CP-final"] = not lo - 1e-12 <= truth <= hi + 1e-12
            widths["CP-final"].append(hi - lo)
            for m, c in cs.items():
                lo2, hi2 = c.interval()
                widths[m].append(hi2 - lo2)
            lo3, hi3 = _wilson(s, n, z)
            widths["Wilson-peek"].append(hi3 - lo3)
            for m in methods:
                marg[m] += miss[m]
                any_miss[m] |= miss[m]
        for m in methods:
            simul[m] += any_miss[m]
    return {m: {"marginal_miss": marg[m] / (reps * k), "simultaneous_miss": simul[m] / reps,
                "mean_final_width": float(np.mean(widths[m]))} for m in methods}, k


def cross_session(cells_a, cells_b, alpha=0.05, boot=2000, seed=1):
    """Final session-A interval (full A data) vs session-B mean. Expected miss from B's own sampling noise alone:
    draw B ~ Binomial(n_B, p_hat_A) boot times and count how often the A interval misses the simulated B mean."""
    keys = sorted(set(cells_a) & set(cells_b))
    k = len(keys)
    a = alpha / k
    rng = np.random.default_rng(seed)
    obs = {"RC-mix": 0, "plug-in": 0}
    exp = {"RC-mix": 0.0, "plug-in": 0.0}
    for key in keys:
        pa = np.array(cells_a[key], float)
        b = np.array(cells_b[key], float)
        for m, cls in (("RC-mix", RCMixCS), ("plug-in", BettingCS)):
            cs = cls(a)
            for x in pa:
                cs.update(float(x))
            lo, hi = cs.interval()
            obs[m] += not lo - 1e-12 <= b.mean() <= hi + 1e-12
            sim = rng.binomial(len(b), pa.mean(), size=boot) / len(b)
            exp[m] += float(np.mean((sim < lo - 1e-12) | (sim > hi + 1e-12)))
    return {m: {"observed_miss": obs[m] / k, "expected_miss_from_B_noise": exp[m] / k, "cells": k} for m in obs}


def rq2(ad_a, ad_b, ia_a, ia_b, reps=200):
    res, md = {}, ["# RQ2: coverage (preregistered)", ""]
    for bench, ca, cb in (("AgentDojo", cells_ad(ad_a), cells_ad(ad_b)),
                          ("InjecAgent ASR-valid", cells_ia(ia_a), cells_ia(ia_b)),
                          ("InjecAgent ASR-all", cells_ia(ia_a, "asr_all"), cells_ia(ia_b, "asr_all"))):
        w, k = within_session(ca, reps=reps)
        x = cross_session(ca, cb)
        res[bench] = {"within_session_A": w, "cross_session": x, "K": k}
        md += [f"## {bench} (K = {k} cells, α = 0.05 simultaneous)", "",
               "| method | marginal time-uniform miss | simultaneous miss | mean final width |", "|---|---|---|---|"]
        md += [f"| {m} | {v['marginal_miss']:.4f} | {v['simultaneous_miss']:.3f} | {v['mean_final_width']:.3f} |" for m, v in w.items()]
        md += ["", "| method | cross-session observed miss | expected from B noise |", "|---|---|---|"]
        md += [f"| {m} | {v['observed_miss']:.3f} | {v['expected_miss_from_B_noise']:.3f} |" for m, v in x.items()]
        md.append("")
    return res, md


if __name__ == "__main__":
    what = sys.argv[1]
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else None
    if what == "c2":
        _write("c2", *c2(load_ad("B"), load_ad("A")))
    elif what == "e3p":
        _write("e3p", *e3p(load_ad("B"), reps=reps or 400))
    elif what == "rq2":
        _write("rq2", *rq2(load_ad("A"), load_ad("B"), load_ia("A"), load_ia("B"), reps=reps or 200))
    else:
        raise SystemExit(what)
