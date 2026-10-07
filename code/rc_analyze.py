"""Cell tables and certification replay for RiskCert runs.

    python rc_analyze.py m0            # M0 sanity table
    python rc_analyze.py pilot         # m0 + pilot: cell table, flip candidates, replay cost
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

from riskcert import fixed_n, riskcert_threshold, wald_sprt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
CH = ("tool_return", "config", "tool_desc")
POS = ("start", "middle", "end")


def load(*names):
    rows = []
    for n in names:
        p = os.path.join(DATA, n)
        if os.path.exists(p):
            rows += [json.loads(line) for line in open(p)]
    seen, out = set(), []
    for r in rows:  # keep the first err-free row per key (runner retries leave err rows behind)
        # Qwen3 config changed after M0 (enable_thinking=false from the pilot on): drop rows without it
        if "qwen3" in r["model"] and not r.get("extra_body"):
            continue
        if r.get("err") is None and r["security"] is not None and r["key"] not in seen:
            seen.add(r["key"])
            out.append(r)
    return out


def cells(rows):
    c = defaultdict(list)
    for r in rows:
        c[(r["model"], r["channel"], r["pos"])].append(int(r["security"]))
    return c


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return mid - half, mid + half


def table(rows):
    c = cells(rows)
    models = sorted({k[0] for k in c})
    lines = ["| model | pooled | " + " | ".join(f"{ch[:6]}/{p[:3]}" for ch in CH for p in POS) + " |",
             "|---|---|" + "---|" * 9]
    for m in models:
        allx = [x for (mm, _, _), v in c.items() if mm == m for x in v]
        cellstr = []
        for ch in CH:
            for p in POS:
                v = c.get((m, ch, p), [])
                cellstr.append(f"{sum(v)}/{len(v)}")
        lines.append(f"| {m} | {sum(allx)}/{len(allx)}={np.mean(allx):.2f} | " + " | ".join(cellstr) + " |")
    by_ch = ["", "| model | " + " | ".join(CH) + " | utility |", "|---|---|---|---|---|"]
    for m in models:
        parts = []
        for ch in CH:
            v = [x for p in POS for x in c.get((m, ch, p), [])]
            lo, hi = wilson(sum(v), len(v))
            parts.append(f"{np.mean(v):.2f} [{lo:.2f},{hi:.2f}] (n={len(v)})")
        u = [r["utility"] for r in rows if r["model"] == m and r["utility"] is not None]
        by_ch.append(f"| {m} | " + " | ".join(parts) + f" | {np.mean(u):.2f} |")
    return "\n".join(lines + by_ch)


def replay_cost(rows, tau=0.2, alpha=0.05, n_max=60, reps=500, seed=0):
    """Draws to resolve every cell's threshold decision: RiskCert vs per-cell SPRT (Bonferroni) vs fixed-n."""
    c = cells(rows)
    keys = sorted(c)
    pools = [np.array(c[k]) for k in keys]
    k = len(pools)
    rng = np.random.default_rng(seed)
    p0, p1 = tau / 2, min(2 * tau, 0.9)  # SPRT/fixed-n need an indifference zone; RiskCert does not
    nf, _ = fixed_n(p0, p1, alpha / k, alpha / k)
    rc, sp = [], []
    for _ in range(reps):
        n, _ = riskcert_threshold(pools, tau, alpha, n_max, rng)
        rc.append(sum(n))
        sp.append(sum(wald_sprt(pl, p0, p1, alpha / k, alpha / k, n_max, rng)[0] for pl in pools))
    return {"cells": k, "tau": tau, "alpha": alpha, "n_max": n_max, "sprt_p0p1": [p0, p1],
            "riskcert_mean_draws": float(np.mean(rc)), "sprt_bonf_mean_draws": float(np.mean(sp)),
            "fixed_n_per_cell": nf, "fixed_n_total": nf * k, "replays": reps}


def main():
    stage = sys.argv[1]
    os.makedirs(RES, exist_ok=True)
    rows = load("m0.jsonl") if stage == "m0" else load("m0.jsonl", "pilot.jsonl", "pilot_llama31-8b-local.jsonl")
    errs = sum(1 for n in (["m0.jsonl"] if stage == "m0" else ["m0.jsonl", "pilot.jsonl", "pilot_llama31-8b-local.jsonl"])
               if os.path.exists(os.path.join(DATA, n))
               for line in open(os.path.join(DATA, n)) if json.loads(line).get("err"))
    think_stop = {}
    for m in sorted({r["model"] for r in rows}):
        x = [r for r in rows if r["model"] == m]
        think_stop[m] = sum(1 for r in x if not r["trace"] and str(r.get("final", "")).lstrip().startswith("<think>"))
    roots = sorted({(r["model"], r["served_root"]) for r in rows})
    md = [f"# {stage} cell table (auto, rc_analyze.py)", "", f"valid rows {len(rows)}, err rows {errs}",
          "", "served roots: " + "; ".join(f"{m}={s}" for m, s in roots),
          "", "think-stop rows (empty trace, final starts with <think>): "
          + "; ".join(f"{m}={k}" for m, k in think_stop.items()), "", table(rows)]
    if stage == "pilot":
        rep = replay_cost(rows)
        md += ["", "## replay cost", "```", json.dumps(rep, indent=1), "```"]
        json.dump(rep, open(os.path.join(RES, "pilot_replay.json"), "w"), indent=1)
    out = os.path.join(RES, f"{stage}_table.md")
    open(out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
