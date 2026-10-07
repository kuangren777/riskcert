"""Sensitivity of the session-B analyses to the 3 Qwen2.5-7B rows that overflowed the 16k context (PM 2026-10-06).

Primary analysis: these keys were rerun until a valid row existed (2 on the first rerun, 1 on the second), which can
favour shorter episodes. Variants here:
  drop  - the 3 keys are missing (their units leave the paired pools)
  violation - the 3 keys count as violations (security = True), the least favourable case
  (a "safe" variant was dropped after the independent check: the 3 rerun rows already have security = False,
   so it was identical to the primary analysis and tested nothing)
Recomputed: C2 certification (no budget curve), RQ2 cross-session coverage for AgentDojo, E3' on real pools (100 reps).
RQ2 within-session coverage uses session-A rows only and is unaffected.

    python3 sessb_sensitivity.py   -> results/sessb_sensitivity.json, .md
"""
import json
import os

import rc_replay
import sessb_analyze as S

HERE = os.path.dirname(os.path.abspath(__file__))


def overflow_keys():
    keys = []
    for line in open(os.path.join(S.DATA, "e2b_qwen25.jsonl")):
        r = json.loads(line)
        if r.get("err") and "context" in r["err"].lower() or (r.get("err") and "maximum" in r["err"]):
            keys.append(r["key"])
    return sorted(set(keys))


def variant(rows, keys, how):
    ks = set(keys)
    if how == "drop":
        return [r for r in rows if r["key"] not in ks]
    return [dict(r, security=True) if r["key"] in ks else r for r in rows]


def main():
    keys = overflow_keys()
    rows_b, rows_a = S.load_ad("B"), S.load_ad("A")
    out = {"overflow_keys": keys, "variants": {}}
    md = ["# Session-B sensitivity: 16k context-overflow rows (PM 2026-10-06)", "", f"keys: {len(keys)}", ""]
    for how in ("primary", "drop", "violation"):
        rb = rows_b if how == "primary" else variant(rows_b, keys, how)
        c2, _ = S.c2(rb, rows_a, budget_reps=0)
        x = S.cross_session(S.cells_ad(rows_a), S.cells_ad(rb))
        tb = rc_replay.pools_from_rows(rb)
        e = rc_replay.peek_experiment(tb, reps=100)["results"]["real"]
        out["variants"][how] = {"c2_r1": c2["certified_r1"], "c2_r1prime": c2["certified_r1prime"], "c2_pass": c2["pass"],
                                "rq2_cross_AD": x, "e3p_real_100reps": e}
        md += [f"## {how}", "", f"C2: R1′ {c2['certified_r1prime']} vs R1 {c2['certified_r1']} → {'PASS' if c2['pass'] else 'FAIL'}",
               f"RQ2 cross-session AgentDojo observed miss: RC-mix {x['RC-mix']['observed_miss']:.3f}, plug-in {x['plug-in']['observed_miss']:.3f}",
               "E3′ real (100 reps) FWER R1/R1′ at L=1,5,20: " + ", ".join(
                   f"{e[L]['R1']['fwer']:.2f}/{e[L]['R1prime']['fwer']:.2f}" for L in ("1", "5", "20")), ""]
    json.dump(out, open(os.path.join(S.RES, "sessb_sensitivity.json"), "w"), indent=1, default=str)
    open(os.path.join(S.RES, "sessb_sensitivity.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
