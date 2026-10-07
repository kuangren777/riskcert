"""X1-X3 positioning experiments (plan.md X1-X3, preregistered a98051bb). CPU replays on existing data.

    python3 x_positioning.py x1 [reps]   warm start under a wrong prior: RiskCert vs AgentAssay warm-start SPRT
    python3 x_positioning.py x2 [reps]   Rank CS / BB-EDGE on given-order fixed grids
    python3 x_positioning.py x3          discovery stop vs Good-Turing and Chao1 coverage rules
"""
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import beta as beta_dist

import e4_replay as E4
import ec_compare as EC
import fixed_grid_sim as FG
import sessb_analyze as S
from riskcert import RCMixCS, prior_theta, three_way

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

# ---------------------------------------------------------------------------------------------- X1
TAU, EPS, CAP, ALPHA = 0.1, 0.05, 600, 0.05
P_LO, P_HI = TAU - EPS, TAU + EPS          # AgentAssay H1 (regression) p = 0.05, H0 (baseline) p = 0.15
K = len(E4.CH) * len(E4.POS)
A_CELL = ALPHA / K
STEP1 = math.log(P_LO / P_HI)              # LLR(H1 vs H0) increment for a violation
STEP0 = math.log((1 - P_LO) / (1 - P_HI))  # ... for a non-violation
UP = math.log((1 - A_CELL) / A_CELL)       # accept H1 ("below tau") when Lambda >= UP (alpha = beta = A_CELL)
DN = math.log(A_CELL / (1 - A_CELL))       # accept H0 ("above tau") when Lambda <= DN


def warm_lambda0(k0, n0):
    """AgentAssay Def 7.8 eq 49: Lambda0 = log B(theta - delta; a0, b0) / B(theta; a0, b0), a0 = k0 + 1, b0 = n0 - k0 + 1."""
    a0, b0 = k0 + 1, n0 - k0 + 1
    return float(beta_dist.logpdf(P_LO, a0, b0) - beta_dist.logpdf(P_HI, a0, b0))


def cell_counts(units):
    c = {}
    for pos in E4.POS:
        us = [v for p, v in units if p == pos]
        for ch in E4.CH:
            c[(ch, pos)] = (sum(v[ch] for v in us), len(us))
    return c


def x1_arms(truth, prev_counts):
    """Hedges (RiskCert) and Lambda0 (AgentAssay) per arm; adversary uses the pool truth to point the wrong way."""
    arms = {}
    for arm in ("cold", "prev", "adversary"):
        th, l0 = {}, {}
        for c, p in truth.items():
            k0, n0 = prev_counts[c]
            if arm == "cold":
                th[c], l0[c] = 0.5, 0.0
            elif arm == "prev":
                th[c], l0[c] = prior_theta(k0 / max(n0, 1), 0.8), warm_lambda0(k0, n0)
            else:
                below = p < TAU
                th[c] = np.full(E4.GRID.shape, 0.8 if below else 0.2)  # 0.8 toward 'p > m' when truth is below
                kw = round(n0 * (P_HI if below else P_LO))
                l0[c] = warm_lambda0(kw, n0)
        arms[arm] = (th, l0)
    return arms


def x1_replay(units, truth, th, l0, seq):
    """One replay of both methods on the same draws. Returns (rc_wrong, rc_runs, aa_wrong, aa_runs)."""
    rc = {c: RCMixCS(A_CELL, theta=th[c]) for c in truth}
    lam = dict(l0)
    rc_dec, aa_dec, rc_n, aa_n = {}, {}, dict.fromkeys(truth, 0), dict.fromkeys(truth, 0)
    for idx in seq:
        pos, v = units[idx]
        for ch in E4.CH:
            c = (ch, pos)
            x = v[ch]
            if c not in rc_dec:
                rc[c].update(float(x))
                rc_n[c] += 1
                d = three_way(*rc[c].interval(), TAU, EPS)
                if d or rc_n[c] >= CAP:
                    rc_dec[c] = d
            if c not in aa_dec:
                lam[c] += STEP1 if x else STEP0
                aa_n[c] += 1
                if lam[c] >= UP:
                    aa_dec[c] = "below"
                elif lam[c] <= DN:
                    aa_dec[c] = "above"
                elif aa_n[c] >= CAP:
                    aa_dec[c] = None
        if len(rc_dec) == K and len(aa_dec) == K:
            break

    def wrong(dec):
        w = 0
        for c, d in dec.items():
            p = truth[c]
            w += (d == "above" and p <= P_LO) or (d == "below" and p >= P_HI) or (d == "near" and abs(p - TAU) >= EPS)
        return w

    return wrong(rc_dec) > 0, sum(rc_n.values()), wrong(aa_dec) > 0, sum(aa_n.values())


def x1_transition(args):
    prev, new, reps, seed = args
    rows_a, rows_b = S.load_ad("A"), S.load_ad("B")
    units = E4.units_of(rows_a + rows_b, new)
    truth, _ = E4.truths(units)
    prev_counts = cell_counts(E4.units_of(rows_a, prev))
    arms = x1_arms(truth, prev_counts)
    rng = np.random.default_rng(seed)
    acc = {a: {"rc_w": 0, "rc_n": [], "aa_w": 0, "aa_n": []} for a in arms}
    for _ in range(reps):
        seq = rng.integers(len(units), size=CAP * 3 * 3)
        for a, (th, l0) in arms.items():
            rw, rn, aw, an = x1_replay(units, truth, th, l0, seq)
            acc[a]["rc_w"] += rw
            acc[a]["aa_w"] += aw
            acc[a]["rc_n"].append(rn)
            acc[a]["aa_n"].append(an)
    return {"prev": prev, "new": new, "reps": reps, "truth": {f"{c[0]}|{c[1]}": p for c, p in truth.items()},
            "arms": {a: {"riskcert_fwer": v["rc_w"] / reps, "agentassay_fwer": v["aa_w"] / reps,
                         "riskcert_mean_runs": float(np.mean(v["rc_n"])), "agentassay_mean_runs": float(np.mean(v["aa_n"]))}
                     for a, v in acc.items()}}


def x1(reps=500):
    jobs = [(p, n, reps, i) for i, (p, n) in enumerate(E4.TRANSITIONS)]
    with ProcessPoolExecutor(4) as ex:
        out = list(ex.map(x1_transition, jobs))
    se = math.sqrt(ALPHA * (1 - ALPHA) / reps)
    lim = ALPHA + 3 * se
    rc_ok = all(t["arms"][a]["riskcert_fwer"] <= lim for t in out for a in t["arms"])
    aa_breaks = any(t["arms"]["adversary"]["agentassay_fwer"] > lim for t in out)
    res = {"limit": lim, "transitions": out, "X1_riskcert_valid": rc_ok, "X1_agentassay_adversary_exceeds": aa_breaks,
           "claim": rc_ok and aa_breaks}
    md = [f"# X1: warm start under a wrong prior (9 cell thresholds, {reps} replays, limit {lim:.3f})", "",
          "| transition | arm | RiskCert FWER | AgentAssay FWER | RiskCert runs | AgentAssay runs |", "|---|---|---|---|---|---|"]
    for t in out:
        for a, v in t["arms"].items():
            md.append(f"| {t['prev']} → {t['new']} | {a} | {v['riskcert_fwer']:.3f} | {v['agentassay_fwer']:.3f} | "
                      f"{v['riskcert_mean_runs']:.0f} | {v['agentassay_mean_runs']:.0f} |")
    md += ["", f"RiskCert valid in every arm: {rc_ok}. AgentAssay adversary exceeds the limit somewhere: {aa_breaks}. Claim: {res['claim']}"]
    _write("x1", res, md)


# ---------------------------------------------------------------------------------------------- X2
def x2_replay(pp, pm, protocol, rng):
    units = FG.unit_order(pp, pm, protocol, rng)
    u = rng.random(len(units))
    d = np.where(u < pp[units], 1, np.where(u < pp[units] + pm[units], -1, 0))
    rk, bb = EC.RankCS(), EC.BBEdge()
    thr = 2 / ALPHA
    rk_any = bb_any = False
    for x in d:
        rk.update(float(x))
        bb.update((float(x) + 1) / 2)
        rk_any |= max(rk.e()) >= thr
        bb_any |= bool(EC.e_holm(list(bb.e()), ALPHA))
    return {"RankCS-any": rk_any, "RankCS-final": max(rk.e()) >= thr,
            "BBEDGE-any": bb_any, "BBEDGE-final": bool(EC.e_holm(list(bb.e()), ALPHA))}


def x2_cell(args):
    name, protocol, reps, seed = args
    rng = np.random.default_rng(seed)
    pp, pm = FG.design(name)
    acc = {}
    for _ in range(reps):
        for k, v in x2_replay(pp, pm, protocol, rng).items():
            acc[k] = acc.get(k, 0) + bool(v)
    return {"design": name, "protocol": protocol, **{k: v / reps for k, v in acc.items()}}


def x2(reps=1000):
    cells = [(n, p, reps, 2030 + 10 * i + j) for i, n in enumerate(("A_mild", "B_extreme", "D_adv"))
             for j, p in enumerate(FG.PROTOCOLS)]
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(x2_cell, cells))
    lim = ALPHA + 3 * math.sqrt(ALPHA * (1 - ALPHA) / reps)
    breaks = any(r[k] > lim for r in rows if r["protocol"] in ("fixed-desc", "fixed-asc") for k in ("RankCS-any", "BBEDGE-any"))
    res = {"reps": reps, "limit": lim, "rows": rows, "X2_claim": breaks}
    cols = ("RankCS-any", "RankCS-final", "BBEDGE-any", "BBEDGE-final")
    md = [f"# X2: leaderboard methods on given-order fixed grids (design average 0, {reps} replays, limit {lim:.3f})", "",
          "| design | protocol | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for r in rows:
        md.append(f"| {r['design']} | {r['protocol']} | " + " | ".join(f"{r[k]:.3f}" for k in cols) + " |")
    md += ["", f"Claim (a leaderboard method exceeds the limit on a sorted grid, any-time read): {breaks}"]
    _write("x2", res, md)


# ---------------------------------------------------------------------------------------------- X3
def chao1(counts):
    s = len(counts)
    f1 = sum(1 for v in counts.values() if v == 1)
    f2 = sum(1 for v in counts.values() if v == 2)
    return s + (f1 * f1 / (2 * f2) if f2 > 0 else f1 * (f1 - 1) / 2)


def stop_times(seq, min_n=50):
    """First n >= min_n meeting the Good-Turing (f1/n <= 0.05) and Chao1-coverage (S/Chao1 >= 0.95) rules."""
    counts, gt, ch = {}, None, None
    for n, sp in enumerate(seq, 1):
        if sp is not None:
            counts[sp] = counts.get(sp, 0) + 1
        if n < min_n:
            continue
        f1 = sum(1 for v in counts.values() if v == 1)
        if gt is None and f1 / n <= 0.05:
            gt = n
        c1 = chao1(counts)
        if ch is None and (c1 == 0 or len(counts) / c1 >= 0.95):
            ch = n
        if gt is not None and ch is not None:
            break
    return gt, ch


def continuation(seq, n0, horizon=300):
    if n0 is None or len(seq) < n0 + horizon:
        return None, None
    seen = {s for s in seq[:n0] if s is not None}
    new = 0
    for s in seq[n0:n0 + horizon]:
        if s is not None and s not in seen:
            new += 1
            seen.add(s)
    return new, new / horizon


def x3():
    import e5_run
    import m3b_analyze as M
    rows = []
    for i, c in enumerate(e5_run.CELLS):
        seq = M._seq(i)
        a = M.analyse_cell(seq)
        rc_stop = a["stop_n"] if a["bound"] is not None and math.isfinite(float(a["bound"])) else None
        gt, ch = stop_times(seq)
        r = {"cell": e5_run.cell_name(c), "draws": len(seq)}
        for name, n0 in (("riskcert", rc_stop), ("good_turing", gt), ("chao1_cov", ch), ("rule50", a["rule50_stop"])):
            new, rate = continuation(seq, n0)
            r[name] = {"stop": n0, "species_at_stop": len({s for s in seq[:n0] if s is not None}) if n0 else None,
                       "new_in_300": new, "rate": rate, "rate_above_0.05": None if rate is None else rate > 0.05}
        rows.append(r)
    res = {"rows": rows}
    md = ["# X3: discovery stop vs coverage estimators (frozen E5 sequences)", "",
          "| cell | rule | stop | species at stop | new in next 300 | rate |", "|---|---|---|---|---|---|"]
    for r in rows:
        for name in ("riskcert", "good_turing", "chao1_cov", "rule50"):
            v = r[name]
            f = lambda x, p=3: "—" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))
            md.append(f"| {r['cell']} | {name} | {f(v['stop'])} | {f(v['species_at_stop'])} | {f(v['new_in_300'])} | {f(v['rate'])} |")
    md += ["", "Only the RiskCert rule carries a certified missing-mass bound (Prop. discovery stop); the other rules are estimates."]
    _write("x3", res, md)


def _write(name, res, md):
    json.dump(res, open(os.path.join(RES, f"{name}.json"), "w"), indent=1, default=str)
    open(os.path.join(RES, f"{name}.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    cmd = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    {"x1": lambda: x1(n or 500), "x2": lambda: x2(n or 1000), "x3": x3}[cmd]()
