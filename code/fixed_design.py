"""Fixed-design certificates (plan.md correction M3c, frozen c1e7a0f8).

Primary: empirical-Bernstein lambda-mixture bound read at the final time N (theory/EB_HETEROGENEOUS.md §6;
Howard et al. 2021 Thm 4). Valid for the design average delta_bar = mean_u E[D_u] of independent units with
heterogeneous means, in any order fixed before the data. Floor: Hoeffding. Sensitivity (descriptive only): the
original SignCS / ContrastCS over 200 random unit orders, numpy seed 2027.

    python3 fixed_design.py   -> results/m3c.json, results/m3c.md
"""
import collections
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES, DATA = os.path.join(HERE, "..", "results"), os.path.join(HERE, "..", "data")
LAMS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEED, PERMS = 2027, 200


def _psi(l):
    return -math.log(1 - l) - l


def eb_halfwidth(d, a):
    """§6: final-time half-width h for D in [-1,1] at level a per tail (two-sided CI has level 1 - 2a)."""
    n, csum, v = len(d), 0.0, 0.0
    for i, x in enumerate(d, 1):
        mhat = csum / i  # predictable centring with one pseudo-observation 0
        v += (x - mhat) ** 2 / 4
        csum += x
    logw = -math.log(len(LAMS))

    def logm(h):
        terms = [logw + l * n * h / 2 - _psi(l) * v for l in LAMS]
        mx = max(terms)
        return mx + math.log(sum(math.exp(t - mx) for t in terms))

    target = math.log(1 / a)
    if logm(2.0) < target:
        return 2.0
    lo, hi = 0.0, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if logm(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def hoeffding_halfwidth(n, a):
    return math.sqrt(2 * math.log(1 / a) / n)


def bound(d, a, method):
    m = sum(d) / len(d)
    h = eb_halfwidth(d, a) if method == "eb" else hoeffding_halfwidth(len(d), a)
    return m - h, m + h


def paired_units(rows, a_ch, b_ch):
    """{model: [D_u in deterministic key order]} for units observed on both channels."""
    u = collections.defaultdict(dict)
    for r in rows:
        k = r["key"].split("|")
        u[(r["model"], "|".join(k[2:]))][r["channel"]] = int(bool(r["security"]))
    out = collections.defaultdict(list)
    for (m, key) in sorted(u):
        v = u[(m, key)]
        if a_ch in v and b_ch in v:
            out[m].append(v[a_ch] - v[b_ch])
    return out


def _first_rows(path):
    seen, rows = set(), []
    for line in open(path):
        r = json.loads(line)
        if r.get("err") is None and r["key"] not in seen:
            seen.add(r["key"])
            rows.append(r)
    return rows


def confirmation(path, alpha=0.05, k=2):
    """C1 / C1b: one-sided H1 config > tool_return (delta = tr - cfg < 0) at a = alpha/k per model."""
    d = paired_units(_first_rows(path), "tool_return", "config")
    out = {}
    for m, ds in d.items():
        r = {"units": len(ds), "mean_delta": sum(ds) / len(ds)}
        for meth in ("eb", "hoeffding"):
            lo, hi = bound(ds, alpha / k, meth)
            r[meth] = {"upper": hi, "certified_cfg_riskier": hi < 0}
        out[m] = r
    return out


def session_b(alpha=0.05):
    """C2 / RQ1: all within-model channel contrasts of session B, two-sided, a = alpha/(2K) per tail."""
    import rc_replay
    import sessb_analyze as S
    t = rc_replay.pools_from_rows(S.load_ad("B"))
    decs = rc_replay.decisions_for(t)
    k = len(decs)
    a = alpha / (2 * k)
    rows = []
    for d in decs:
        keys = sorted(t[d[0]])
        ds = [t[d[0]][kk][d[1]] - t[d[0]][kk][d[2]] for kk in keys]
        r = {"model": d[0], "a": d[1], "b": d[2], "n": len(ds), "delta": sum(ds) / len(ds)}
        for meth in ("eb", "hoeffding"):
            lo, hi = bound(ds, a, meth)
            r[meth] = {"lo": lo, "hi": hi, "sign": ">" if lo > 0 else "<" if hi < 0 else None}
        rows.append(r)
    return {"K": k, "a_per_tail": a, "rows": rows}


def sensitivity_orders(alpha=0.05):
    """Descriptive only: original R1' (SignCS) certified count on session B over PERMS random unit orders."""
    import rc_replay
    import sessb_analyze as S
    from riskcert import BettingCS, SignCS, sign_decision
    t = rc_replay.pools_from_rows(S.load_ad("B"))
    decs = rc_replay.decisions_for(t)
    k = len(decs)
    rng = np.random.default_rng(SEED)
    counts = []
    for _ in range(PERMS):
        c = 0
        for d in decs:
            us = list(t[d[0]].values())
            order = rng.permutation(len(us))
            sg = SignCS(alpha / k, cls=BettingCS)
            for i in order:
                sg.update(us[i][d[1]], us[i][d[2]])
            c += sign_decision(*sg.interval()) is not None
        counts.append(c)
    return {"perms": PERMS, "seed": SEED, "median": float(np.median(counts)), "min": int(min(counts)), "max": int(max(counts))}


def main():
    res = {"C1": confirmation(os.path.join(DATA, "confirm_c1.jsonl")),
           "C1b": confirmation(os.path.join(DATA, "confirm_c1b.jsonl")),
           "sessionB": session_b(), "sensitivity_R1prime_orders": sensitivity_orders()}
    sb = res["sessionB"]
    for meth in ("eb", "hoeffding"):
        sb[f"certified_{meth}"] = sum(r[meth]["sign"] is not None for r in sb["rows"])
    md = ["# M3c fixed-design certificates (primary EB mixture, floor Hoeffding)", ""]
    for name in ("C1", "C1b"):
        md += [f"## {name} (one-sided, config riskier, a = 0.025 per model)", "",
               "| model | units | mean δ (tr − cfg) | EB upper | EB certified | Hoeffding upper | Hoeffding certified |", "|---|---|---|---|---|---|---|"]
        for m, r in res[name].items():
            md.append(f"| {m} | {r['units']} | {r['mean_delta']:+.3f} | {r['eb']['upper']:+.3f} | {r['eb']['certified_cfg_riskier']} | "
                      f"{r['hoeffding']['upper']:+.3f} | {r['hoeffding']['certified_cfg_riskier']} |")
        md.append("")
    md += [f"## Session B channel contrasts (two-sided, K = {sb['K']})", "",
           f"certified: EB {sb['certified_eb']} / {sb['K']}, Hoeffding {sb['certified_hoeffding']} / {sb['K']}", "",
           "| model | a − b | n | δ | EB CI | EB | Hoeffding |", "|---|---|---|---|---|---|---|"]
    for r in sb["rows"]:
        md.append(f"| {r['model']} | {r['a']} − {r['b']} | {r['n']} | {r['delta']:+.3f} | [{r['eb']['lo']:+.3f}, {r['eb']['hi']:+.3f}] | "
                  f"{r['eb']['sign'] or '?'} | {r['hoeffding']['sign'] or '?'} |")
    s = res["sensitivity_R1prime_orders"]
    md += ["", f"Descriptive only (no claim): original R1′ on session B over {s['perms']} random orders (seed {s['seed']}): "
           f"median {s['median']}, range {s['min']}–{s['max']} certified of {sb['K']}."]
    json.dump(res, open(os.path.join(RES, "m3c.json"), "w"), indent=1)
    open(os.path.join(RES, "m3c.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
