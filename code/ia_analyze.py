"""InjecAgent E1 analysis: per-stratum tables (ASR-valid and ASR-all, invalid as its own class) and paired
setting contrasts (enhanced - base, per attack) certified with the R1 contrast CS (plan.md §2, PM ruling 2026-10-06).

ASR-valid = succ / (succ + unsucc); ASR-all = succ / all (invalid = no violation). A paired contrast under ASR-valid
uses only units whose outputs are valid in both settings (estimand conditioned on validity).

    python3 ia_analyze.py e1   -> results/e1_ia_table.md, results/e1_ia.json
"""
import collections
import glob
import json
import os
import sys

import numpy as np

from riskcert import BettingCS, ContrastCS, RCMixCS, SignCS, sign_decision

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES = os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "results")
ALPHA = 0.05


def load(stage):
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, f"ia_{stage}_*.jsonl"))):
        rows += [json.loads(line) for line in open(f)]
    seen, out = set(), []
    for r in rows:
        if r.get("err") is None and r["key"] not in seen:
            seen.add(r["key"])
            out.append(r)
    return out


def cell_stats(rs):
    c = collections.Counter(r["eval"] for r in rs)
    n, s, u, i = len(rs), c["succ"], c["unsucc"], c["invalid"]
    return {"n": n, "succ": s, "unsucc": u, "invalid": i, "asr_all": s / n if n else None,
            "asr_valid": s / (s + u) if s + u else None, "invalid_rate": i / n if n else None}


def cs_interval(xs, alpha):
    cs = RCMixCS(alpha)
    for x in xs:
        cs.update(float(x))
    return cs.interval()


def contrasts(rows, alpha=ALPHA):
    """enhanced - base per (model, attack), paired by (case, rep); both metrics; one family over all of them."""
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["model"], r["attack"])][(r["case"], r["rep"], r["setting"])] = r["eval"]
    fams = sorted(by)
    k = len(fams)
    out = {}
    for metric in ("asr_valid", "asr_all"):
        for fam in fams:
            ev = by[fam]
            units = sorted({(c, rp) for c, rp, _ in ev})
            pairs = []
            for u in units:
                b, e = ev.get(u + ("base",)), ev.get(u + ("enhanced",))
                if b is None or e is None:
                    continue
                if metric == "asr_valid" and "invalid" in (b, e):
                    continue
                pairs.append((int(e == "succ"), int(b == "succ")))
            c, sg = ContrastCS(alpha / k), SignCS(alpha / k, cls=BettingCS)
            for xe, xb in pairs:
                c.update(xe, xb)
                sg.update(xe, xb)
            lo, hi = c.interval()
            slo, shi = sg.interval()
            d = float(np.mean([xe - xb for xe, xb in pairs])) if pairs else None
            disc = sum(xe != xb for xe, xb in pairs)
            out[(metric,) + fam] = {"n_pairs": len(pairs), "n_discordant": disc, "delta": d, "cs": [lo, hi],
                                    "sign_contrast": sign_decision(lo, hi), "sign_scale_cs": [slo, shi],
                                    "sign": sign_decision(slo, shi)}
    return out, k


def reversals(con, metric):
    by = collections.defaultdict(dict)
    for (m_, model, attack), v in con.items():
        if m_ == metric and v["sign"] in (">", "<"):
            by[attack][model] = v["sign"]
    revs = []
    for attack, ms in by.items():
        ups = [m for m, s in ms.items() if s == ">"]
        dns = [m for m, s in ms.items() if s == "<"]
        revs += [(attack, a, b) for a in ups for b in dns]
    return revs


def main(stage):
    rows = load(stage)
    models = sorted({r["model"] for r in rows})
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["attack"], r["setting"])].append(r)
    con, k = contrasts(rows)
    md = [f"# InjecAgent {stage} (auto, ia_analyze.py)", "",
          f"valid rows {len(rows)}; models {len(models)}; contrast family K={k} per metric, alpha={ALPHA}", "",
          "Per stratum: ASR-valid = succ/(succ+unsucc) | ASR-all = succ/n | invalid rate. n per cell in brackets.", "",
          "| model | dh/base | dh/enhanced | ds/base | ds/enhanced | pooled ASR-all | pooled invalid |", "|---|---|---|---|---|---|---|"]
    js = {"cells": {}, "contrasts": {}, "pooled": {}}
    for m in models:
        line = [m]
        for a in ("dh", "ds"):
            for s in ("base", "enhanced"):
                st = cell_stats(cells[(m, a, s)])
                js["cells"][f"{m}|{a}|{s}"] = st
                av = "—" if st["asr_valid"] is None else f"{st['asr_valid']:.2f}"
                line.append(f"{av} \\| {st['asr_all']:.2f} \\| {st['invalid_rate']:.2f} ({st['n']})")
        pooled = cell_stats([r for r in rows if r["model"] == m])
        js["pooled"][m] = pooled
        line += [f"{pooled['succ']}/{pooled['n']}={pooled['asr_all']:.2f}", f"{pooled['invalid_rate']:.2f}"]
        md.append("| " + " | ".join(line) + " |")
    for metric in ("asr_valid", "asr_all"):
        md += ["", f"## Setting contrasts enhanced − base, {metric} (R1: RC-mix CS on δ; R1': plug-in CS on discordant pairs; simultaneous alpha={ALPHA}; reversals use R1')", "",
               "| model | attack | n pairs | discordant | δ | R1 CS on δ | R1 | R1' CS on 2q−1 | R1' |", "|---|---|---|---|---|---|---|---|---|"]
        for (m_, model, attack), v in sorted(con.items()):
            if m_ != metric:
                continue
            js["contrasts"][f"{metric}|{model}|{attack}"] = v
            d = "—" if v["delta"] is None else f"{v['delta']:+.3f}"
            md.append(f"| {model} | {attack} | {v['n_pairs']} | {v['n_discordant']} | {d} | [{v['cs'][0]:+.3f}, {v['cs'][1]:+.3f}] | "
                      f"{v['sign_contrast'] or '?'} | [{v['sign_scale_cs'][0]:+.3f}, {v['sign_scale_cs'][1]:+.3f}] | {v['sign'] or '?'} |")
        rv = reversals(con, metric)
        js[f"reversals_{metric}"] = rv
        md += ["", f"certified reversals ({metric}): {len(rv)} " + "; ".join(f"{a}: {x} > vs {y} <" for a, x, y in rv)]
    open(os.path.join(RES, f"{stage}_ia_table.md"), "w").write("\n".join(md) + "\n")
    json.dump(js, open(os.path.join(RES, f"{stage}_ia.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "e1")
