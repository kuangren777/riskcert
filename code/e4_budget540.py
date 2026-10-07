"""EXPLORATORY (added after the runs, PM ruling 2026-10-06; appendix only): decisions the E4 replay closes within the
same 540-run budget as each live arm, for the two transitions that have live runs. CPU only, no new calls.

    python3 e4_budget540.py   -> results/e4_budget540.json
"""
import json
import os

import numpy as np

import e4_replay as E
import sessb_analyze as S

BUDGET, REPS = 540, 200


def run_arm_budget(units, seq, th_c, th_s, tc, tcon, budget=BUDGET):
    """Same procedure as e4_replay.run_arm, stopped once the channel-run cost reaches the budget."""
    from riskcert import BettingCS, RCMixCS, SignCS, sign_decision, three_way
    cells = {c: RCMixCS(E.ALPHA / E.K, theta=th_c[c]) for c in tc}
    signs = {ab: SignCS(E.ALPHA / E.K, cls=BettingCS, theta=th_s[ab]) for ab in E.CONTRASTS}
    out, cost = {}, 0
    for idx in seq:
        pos, v = units[idx]
        need = {ch for ch in E.CH if ("cell", ch, pos) not in out}
        for ab in E.CONTRASTS:
            if ("sign",) + ab not in out:
                need |= set(ab)
        if cost + len(need) > budget or len(out) == E.K:
            break
        cost += len(need)
        for ch in need:
            key = ("cell", ch, pos)
            if key in out:
                continue
            cs = cells[(ch, pos)]
            cs.update(float(v[ch]))
            d = three_way(*cs.interval(), E.TAU, E.EPS)
            if d or cs.n >= E.CAP:
                out[key] = d
        for ab in E.CONTRASTS:
            key = ("sign",) + ab
            if key in out:
                continue
            sg = signs[ab]
            sg.update(float(v[ab[0]]), float(v[ab[1]]))
            d = sign_decision(*sg.interval())
            if d:
                out[key] = d
    return sum(d is not None for d in out.values())


def main():
    rows_a, rows_b = S.load_ad("A"), S.load_ad("B")
    live = {"gpt-5.4-nano-2026-03-17": "e4_live_gpt-5.4-nano-2026-03-17.json", "qwen3-32b-local": "e4_live_qwen3-32b-local.json"}
    res = {}
    for prev, new in (E.TRANSITIONS[1], E.TRANSITIONS[3]):
        units = E.units_of(rows_a + rows_b, new)
        tc, tcon = E.truths(units)
        pc, pcon = E.truths(E.units_of(rows_a, prev))
        rng = np.random.default_rng(0)
        dec = {a: [] for a in E.ARMS}
        for _ in range(REPS):
            seq = rng.integers(len(units), size=BUDGET)
            for a in E.ARMS:
                dec[a].append(run_arm_budget(units, seq, *E.thetas(a, tc, tcon, pc, pcon), tc, tcon))
        lv = {x["arm"]: x["decided"] for x in json.load(open(os.path.join(S.RES, live[new])))}
        res[new] = {a: {"replay_mean_decided": float(np.mean(dec[a])), "replay_p10_p90": [float(np.percentile(dec[a], 10)), float(np.percentile(dec[a], 90))],
                        "live_decided": lv.get(a)} for a in E.ARMS}
        print(new, res[new])
    json.dump({"exploratory": True, "budget": BUDGET, "reps": REPS, "results": res}, open(os.path.join(S.RES, "e4_budget540.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
