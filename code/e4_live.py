"""E4 live (secondary, approved): the newest release of a chain is re-audited live in each warm-start arm.
Preregistered in plan.md M3b (frozen 2fa8a81d). Purpose: check that live cost matches the replay cost distribution.

Units: (pair from the 15 E1 pairs, position) i.i.d. with replacement, seed 41, the same sequence for every arm.
Each round runs the channels the open decisions need at that unit (open-strata, as in e4_replay.run_arm).
Cap: 180 rounds per arm (<= 540 runs). Arms run one after another. Batches run in parallel and are processed in
round order; channel runs a closed decision no longer needed are recorded with used=false and are not counted.

    python3 e4_live.py <model> <prev_model> [arm ...]   -> data/e4_live_<model>_<arm>.jsonl, results/e4_live_<model>.json
"""
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import e4_replay as E4
import rc_run
import sessb_analyze as S
from riskcert import BettingCS, RCMixCS, SignCS, sign_decision, three_way

HERE = os.path.dirname(os.path.abspath(__file__))
ROUNDS, SEED = 180, 41
# rounds per parallel batch; processing stays in round order, so decisions and cost do not depend on it. A larger
# batch only adds unused over-run rows (fine on a local GPU, PM 2026-10-06: 32B at batch 16)
BATCH = int(os.environ.get("E4_LIVE_BATCH", "4"))


def unit_seq():
    pairs = rc_run.grid_pairs()
    rng = random.Random(SEED)
    return [(pairs[rng.randrange(len(pairs))], E4.POS[rng.randrange(3)]) for _ in range(ROUNDS)]


def run_live(model, prev, arm):
    rows_a, rows_b = S.load_ad("A"), S.load_ad("B")
    tc, tcon = E4.truths(E4.units_of(rows_a + rows_b, model))  # pool estimates: used only by oracle/adversary arms
    pc, pcon = E4.truths(E4.units_of(rows_a, prev))
    th_c, th_s = E4.thetas(arm, tc, tcon, pc, pcon)
    cells = {c: RCMixCS(E4.ALPHA / E4.K, theta=th_c[c]) for c in tc}
    signs = {ab: SignCS(E4.ALPHA / E4.K, cls=BettingCS, theta=th_s[ab]) for ab in E4.CONTRASTS}
    out_path = os.path.join(S.DATA, f"e4_live_{model}_{arm}.jsonl")
    rc_run.check_roots([model])
    gate, lock = rc_run.Gate(model), threading.Lock()
    seq = unit_seq()
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            if r.get("err") is None:
                done[(r["round"], r["channel"])] = r
    out, cost = {}, 0

    def need_at(pos):
        n = {ch for ch in E4.CH if ("cell", ch, pos) not in out}
        for ab in E4.CONTRASTS:
            if ("sign",) + ab not in out:
                n |= set(ab)
        return n

    def go(rk, ch):
        (s, u, i), pos = seq[rk]
        txt = rc_run.positions(s, i)[pos]
        with gate:
            r = rc_run.harness.run_pair(model=model, suite_name=s, user_task_id=u, injection_task_id=i,
                                        injection_text=txt, channel=ch, tag="rc_e4_live")
        r.update(round=rk, arm=arm, pos=pos, key=f"{model}|e4live|{arm}|{ch}|{pos}|{s}|{u}|{i}|{rk}",
                 served_root=rc_run._ROOT[model], backend=rc_run.backend_of(model))
        return r

    rk = 0
    with ThreadPoolExecutor(min(16, gate.burst)) as ex:
        while rk < ROUNDS and len(out) < E4.K:
            batch = list(range(rk, min(rk + BATCH, ROUNDS)))
            jobs = [(b, ch) for b in batch for ch in need_at(seq[b][1]) if (b, ch) not in done]
            for (b, ch), r in zip(jobs, ex.map(lambda j: go(*j), jobs)):
                for _ in range(2):
                    if r.get("err") is None:
                        break
                    r = go(b, ch)
                done[(b, ch)] = r
                with lock, open(out_path, "a") as f:  # append every completed run at once (spend recorded, resumable)
                    f.write(json.dumps(r, default=str) + "\n")
            for b in batch:  # process in round order with the need-set at processing time
                if len(out) == E4.K:
                    break
                pos = seq[b][1]
                need = need_at(pos)
                xs = {}
                for ch in need:
                    r = done.get((b, ch))
                    if r is None:  # needed now but not run in this batch (decision reopened cannot happen; guard)
                        r = go(b, ch)
                        done[(b, ch)] = r
                        with lock, open(out_path, "a") as f:
                            f.write(json.dumps(r, default=str) + "\n")
                    if r.get("err") is not None:
                        raise SystemExit(f"round {b} {ch} failed: {r['err'][:200]}")
                    xs[ch] = float(bool(r["security"]))
                    r["used"] = True
                cost += len(need)
                for ch in E4.CH:
                    key = ("cell", ch, pos)
                    if key in out or ch not in xs:
                        continue
                    cs = cells[(ch, pos)]
                    cs.update(xs[ch])
                    d = three_way(*cs.interval(), E4.TAU, E4.EPS)
                    if d or cs.n >= E4.CAP:
                        out[key] = d
                for ab in E4.CONTRASTS:
                    key = ("sign",) + ab
                    if key in out or not set(ab) <= set(xs):
                        continue
                    sg = signs[ab]
                    sg.update(xs[ab[0]], xs[ab[1]])
                    d = sign_decision(*sg.interval())
                    if d or sg.n >= E4.CAP:
                        out[key] = d
            rk = batch[-1] + 1
    used = sorted(k for k, r in done.items() if r.get("used"))
    json.dump({"used": [list(k) for k in used]}, open(out_path.replace(".jsonl", ".used.json"), "w"))
    return {"arm": arm, "cost": cost, "rounds": rk, "decided": sum(d is not None for d in out.values()),
            "decisions": {"|".join(k): d for k, d in out.items()}}


if __name__ == "__main__":
    model, prev = sys.argv[1], sys.argv[2]
    arms = sys.argv[3:] or list(E4.ARMS)
    res = [run_live(model, prev, a) for a in arms]
    json.dump(res, open(os.path.join(S.RES, f"e4_live_{model}.json"), "w"), indent=1)
    print(json.dumps(res)[:2000])
