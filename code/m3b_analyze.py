"""Preregistered M3b analyses (plan.md M3b, frozen 2fa8a81d). Run once after E4 replay and E5 complete.

  e4   pass rules E4-V (FWER <= 0.079 in every transition x arm), E4-C (oracle <= 0.9 x cold in all 4 transitions AND
       prev <= cold in >= 3 of 4), E4-A descriptive (adversary / cold cost ratio).
  e5   per cell: stop time, certified bound B_c, species, new species in 300 continuation runs r_c; consistent iff
       r_c <= q_{1-0.05/6}(Bin(300, B_c)) / 300; pass iff 6/6. Comparators: Good-Turing f1/n at the stop, and the
       uncalibrated "50 runs with nothing new" rule (its stop time and its new-species rate over the next 300 draws).

    python3 m3b_analyze.py e4 | e5   -> results/e4.json/.md, results/e5.json/.md
"""
import collections
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.stats import binom

import e5_run

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
FWER_MAX = 0.079


def e4(results):
    md = ["# E4: release re-audit warm start (preregistered, replay)", "",
          "| transition | units | prev prior right (of 12) | arm | mean cost | median cost | FWER | mean decided |", "|---|---|---|---|---|---|---|---|"]
    v_ok, c_oracle, c_prev = True, 0, 0
    for t in results:
        a = t["arms"]
        for arm in ("cold", "prev", "oracle", "adversary"):
            x = a[arm]
            v_ok &= x["fwer"] <= FWER_MAX
            md.append(f"| {t['prev']} → {t['new']} | {t['units']} | {t['prev_prior_right']} | {arm} | {x['mean_cost']:.0f} | "
                      f"{x['median_cost']:.0f} | {x['fwer']:.3f} | {x['mean_decided']:.1f} |")
        c_oracle += a["oracle"]["mean_cost"] <= 0.9 * a["cold"]["mean_cost"]
        c_prev += a["prev"]["mean_cost"] <= a["cold"]["mean_cost"]
    n = len(results)
    c_ok = c_oracle == n and c_prev >= 3
    ratios = {f"{t['prev']}→{t['new']}": t["arms"]["adversary"]["mean_cost"] / t["arms"]["cold"]["mean_cost"] for t in results}
    md += ["", f"E4-V validity (FWER ≤ {FWER_MAX} everywhere): **{'PASS' if v_ok else 'FAIL'}**",
           f"E4-C cost claim (oracle ≤ 0.9×cold in {c_oracle}/{n}, prev ≤ cold in {c_prev}/{n}): **{'CLAIM' if c_ok else 'NO CLAIM'}**",
           "E4-A adversary/cold cost ratio: " + ", ".join(f"{k} {v:.2f}" for k, v in ratios.items())]
    return {"E4_V": v_ok, "E4_C": c_ok, "oracle_hits": c_oracle, "prev_hits": c_prev, "adv_ratio": ratios}, md


def _seq(cell_idx):
    rows = {}
    for line in open(os.path.join(DATA, f"e5_{e5_run.cell_name(e5_run.CELLS[cell_idx])}.jsonl")):
        r = json.loads(line)
        if r.get("err") is None:
            rows.setdefault(r["draw"], r)
    out = []
    for i in range(len(rows)):
        if i not in rows:
            break
        out.append(e5_run.species(rows[i]))
    return out


def analyse_cell(sp_seq, delta_c=e5_run.DELTA / len(e5_run.CELLS), cont=e5_run.CONT):
    d = e5_run.Discovery(delta_c=delta_c)
    for sp in sp_seq:
        d.feed(sp)
        if d.stop_n is not None and d.cont_runs >= cont:
            break
    n0 = d.stop_n
    seen_at_stop = collections.Counter(s for s in sp_seq[:n0] if s is not None) if n0 else collections.Counter()
    f1 = sum(1 for v in seen_at_stop.values() if v == 1)
    q = binom.ppf(1 - 0.05 / len(e5_run.CELLS), cont, min(d.bound, 1.0)) if d.bound is not None and math.isfinite(d.bound) else cont
    r = d.new_in_cont / cont if d.cont_runs >= cont else None
    # uncalibrated comparator: stop at the first n with 50 draws since the last new species
    seen, last_new, n50 = set(), 0, None
    for i, s in enumerate(sp_seq, 1):
        if s is not None and s not in seen:
            seen.add(s)
            last_new = i
        if i - last_new >= 50:
            n50 = i
            break
    r50 = None
    if n50 is not None and len(sp_seq) >= n50 + cont:
        seen50 = {s for s in sp_seq[:n50] if s is not None}
        new = 0
        for s in sp_seq[n50:n50 + cont]:
            if s is not None and s not in seen50:
                new += 1
                seen50.add(s)
        r50 = new / cont
    return {"stop_n": n0, "bound": d.bound, "species_at_stop": len(seen_at_stop), "new_in_cont": d.new_in_cont,
            "r": r, "q_limit": q / cont, "consistent": r is not None and r <= q / cont,
            "good_turing": f1 / n0 if n0 else None, "rule50_stop": n50, "rule50_r": r50}


def e5():
    res, md = {}, ["# E5: discovery stop (preregistered)", "",
                   "| cell | stop n | bound B_c | species at stop | new in 300 | r_c | limit q/300 | consistent | Good–Turing f1/n | 50-rule stop | 50-rule r |",
                   "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(e5_run.CELLS):
        a = analyse_cell(_seq(i))
        res[e5_run.cell_name(c)] = a
        f = lambda v, p=3: "—" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
        md.append(f"| {' · '.join(c)} | {a['stop_n']} | {f(a['bound'])} | {a['species_at_stop']} | {a['new_in_cont']} | {f(a['r'])} | "
                  f"{f(a['q_limit'])} | {'yes' if a['consistent'] else 'no'} | {f(a['good_turing'])} | {f(a['rule50_stop'])} | {f(a['rule50_r'])} |")
    ok = all(a["consistent"] for a in res.values())
    stopped = [a for a in res.values() if a["bound"] is not None and math.isfinite(float(a["bound"]))]
    capped = [a for a in res.values() if a not in stopped]
    md += ["", f"**Preregistered rule: {'met' if ok else 'not met'}.** {sum(a['consistent'] for a in stopped)} of {len(stopped)} "
           f"cells that stopped with a finite bound are consistent (substantive test). {len(capped)} cells reached the cap "
           f"with new species still appearing and certified no bound, so their consistency is vacuous. Do not report this as '6/6'."]
    return {"cells": res, "pass": ok}, md


def _write(name, res, md):
    json.dump(res, open(os.path.join(RES, f"{name}.json"), "w"), indent=1, default=str)
    open(os.path.join(RES, f"{name}.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    if sys.argv[1] == "e4":
        parts = []
        for i in range(4):
            parts += json.load(open(os.path.join(RES, f"e4_replay_{i}.json")))
        _write("e4", *e4(parts))
    elif sys.argv[1] == "e5":
        _write("e5", *e5())
