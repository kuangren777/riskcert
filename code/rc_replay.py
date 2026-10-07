"""Contrast-decision replay (E3 core): RiskCert R1+R2+R3 vs paired SPRT vs fixed-n on real outcome pools.

A pool is table[model][unit] = {stratum: 0/1}, complete units only (every stratum observed on the unit).
Each round draws a unit uniformly with replacement and reads the real outcomes of the strata it runs, so
paired structure is kept (THEORY_v2 §0: fresh unit per round). Truth = pool mean.

Decisions are channel-rank contrasts (model, a, b): sign of delta = p_a - p_b. All decisions form one family
of size K; every procedure spends alpha/K per decision.

    python3 rc_replay.py pilot   -> results/pilot_contrast_replay.json
"""
import collections
import json
import math
import os
import sys

import numpy as np
from scipy.stats import binomtest

from riskcert import BettingCS, ContrastCS, RCMixCS, SignCS, paired_sprt_step, sign_decision

HERE = os.path.dirname(os.path.abspath(__file__))


def pools_from_rows(rows, strata=("tool_return", "config", "tool_desc")):
    """Unit = key minus model and channel (pos|suite|ut|it|rep): position is part of the unit draw."""
    t = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        k = r["key"].split("|")
        t[r["model"]]["|".join(k[2:])][r["channel"]] = int(bool(r["security"]))
    return {m: {u: v for u, v in us.items() if all(s in v for s in strata)} for m, us in t.items()}


def decisions_for(table, strata=("tool_return", "config", "tool_desc")):
    pairs = [(a, b) for i, a in enumerate(strata) for b in strata[i + 1:]]
    return [(m, a, b) for m in sorted(table) for a, b in pairs]


def truth(table, d):
    m, a, b = d
    us = table[m].values()
    return float(np.mean([u[a] - u[b] for u in us]))


class _Draw:
    def __init__(self, table, rng):
        self.units = {m: list(us.values()) for m, us in table.items()}
        self.rng = rng

    def __call__(self, m):
        us = self.units[m]
        return us[self.rng.integers(len(us))]


def riskcert(table, decs, alpha, rng, policy="open_strata", eps=0.0, n_max=2000, cs_cls=RCMixCS, stat="contrast"):
    """R1 contrast CS per decision at alpha/K; R2 'tie' futility when eps > 0; R3 allocation policy:
    'pairs'       dedicated pair block per open decision, round robin, cost 2 per draw (Prop 5-A);
    'open_strata' per model one unit per round running every stratum still in an open decision (Prop 5-B)."""
    k, draw = len(decs), _Draw(table, rng)
    mk = (lambda: ContrastCS(alpha / k, cls=cs_cls)) if stat == "contrast" else (lambda: SignCS(alpha / k, cls=cs_cls))
    cs = {d: mk() for d in decs}
    if stat == "sign":
        eps = 0.0  # the sign scale (2q - 1) is not delta, so the delta-tie rule does not apply
    out, cost = {}, 0
    closed_at = {}
    while len(out) < k:
        if policy == "pairs":
            for d in [d for d in decs if d not in out]:
                u = draw(d[0])
                cost += 2
                cs[d].update(u[d[1]], u[d[2]])
                _close(d, cs[d], out, eps, n_max, closed_at, cost)
        elif policy == "open_strata":
            for m in sorted({d[0] for d in decs if d not in out}):
                open_d = [d for d in decs if d[0] == m and d not in out]
                strata = {s for d in open_d for s in d[1:]}
                u = draw(m)
                cost += len(strata)
                for d in open_d:
                    cs[d].update(u[d[1]], u[d[2]])
                    _close(d, cs[d], out, eps, n_max, closed_at, cost)
        else:
            raise ValueError(policy)
    riskcert.closed_at = closed_at  # cumulative run count at which each decision closed (budget curves)
    return cost, out


def _close(d, c, out, eps, n_max, closed_at=None, cost=0):
    s = sign_decision(*c.interval(), eps=eps)
    if s or c.n >= n_max:
        out[d] = s
        if closed_at is not None:
            closed_at[d] = cost


def paired_sprt(table, decs, alpha, rng, eta=0.2, n_max=2000):
    """AgentAssay-style Wald SPRT per decision on discordant pairs, alpha=beta=alpha/K, dedicated pair blocks."""
    k, draw = len(decs), _Draw(table, rng)
    a_ = b_ = alpha / k
    hi, lo = math.log((1 - b_) / a_), math.log(b_ / (1 - a_))
    out, cost = {}, 0
    for d in decs:
        llr, n, s = 0.0, 0, None
        while n < n_max:
            u = draw(d[0])
            n += 1
            cost += 2
            llr = paired_sprt_step(llr, u[d[1]], u[d[2]], eta)
            if llr >= hi:
                s = ">"
                break
            if llr <= lo:
                s = "<"
                break
        out[d] = s
    return cost, out


def paired_sprt_rr(table, decs, alpha, rng, eta=0.2, n_max=2000):
    """Same SPRT, decisions interleaved round robin so a budget curve can be read (closure costs recorded)."""
    k, draw = len(decs), _Draw(table, rng)
    a_ = b_ = alpha / k
    hi, lo = math.log((1 - b_) / a_), math.log(b_ / (1 - a_))
    llr, n, out, cost, closed_at = {d: 0.0 for d in decs}, {d: 0 for d in decs}, {}, 0, {}
    while len(out) < k:
        for d in [d for d in decs if d not in out]:
            u = draw(d[0])
            n[d] += 1
            cost += 2
            llr[d] = paired_sprt_step(llr[d], u[d[1]], u[d[2]], eta)
            s = ">" if llr[d] >= hi else "<" if llr[d] <= lo else None
            if s or n[d] >= n_max:
                out[d], closed_at[d] = s, cost
    paired_sprt_rr.closed_at = closed_at
    return cost, out


def fixed_n(table, decs, alpha, rng, n=100):
    """Fixed design: n units per model running the full table (cost = n x #strata per model), then an exact
    two-sided McNemar (sign) test per decision at alpha/K."""
    k, draw = len(decs), _Draw(table, rng)
    out, cost = {}, 0
    for m in sorted({d[0] for d in decs}):
        strata = {s for d in decs if d[0] == m for s in d[1:]}
        us = [draw(m) for _ in range(n)]
        cost += n * len(strata)
        for d in [d for d in decs if d[0] == m]:
            up = sum(u[d[1]] > u[d[2]] for u in us)
            dn = sum(u[d[1]] < u[d[2]] for u in us)
            s = None
            if up + dn and binomtest(up, up + dn, 0.5).pvalue <= alpha / k:
                s = ">" if up > dn else "<"
            out[d] = s
    return cost, out


def score(out, tru):
    """wrong = certified sign opposite to the pool truth (any sign is wrong when truth is exactly 0)."""
    wrong = sum(1 for d, s in out.items() if s in (">", "<") and not ((s == ">") == (tru[d] > 0) and tru[d] != 0))
    right = sum(1 for d, s in out.items() if s in (">", "<")) - wrong
    return wrong, right


def reversals(out, decs, tru=None):
    """Certified generation/model reversals: same (a, b), two models, opposite certified signs (Cor 1-P).
    With `tru`, only signs that agree with the truth count (correct reversals)."""
    by = collections.defaultdict(dict)
    for d in decs:
        if out.get(d) in (">", "<") and (tru is None or (out[d] == ">") == (tru[d] > 0) and tru[d] != 0):
            by[d[1:]][d[0]] = out[d]
    n = 0
    for ab, ms in by.items():
        signs = list(ms.values())
        n += signs.count(">") * signs.count("<")
    return n


def sweep(table, reps=100, seed=0, n_max=2000):
    decs = decisions_for(table)
    tru = {d: truth(table, d) for d in decs}
    arms = []
    for a in (0.01, 0.05, 0.2):
        for pol in ("pairs", "open_strata"):
            arms.append((f"riskcert/{pol}/a={a}", lambda rng, a=a, pol=pol: riskcert(table, decs, a, rng, pol, n_max=n_max)))
        arms.append((f"riskcert/open_strata+tie0.05/a={a}",
                     lambda rng, a=a: riskcert(table, decs, a, rng, "open_strata", eps=0.05, n_max=n_max)))
        arms.append((f"riskcert_plugin/open_strata/a={a}",
                     lambda rng, a=a: riskcert(table, decs, a, rng, "open_strata", n_max=n_max, cs_cls=BettingCS)))
        for eta in (0.1, 0.2, 0.3):
            arms.append((f"sprt/eta={eta}/a={a}", lambda rng, a=a, eta=eta: paired_sprt(table, decs, a, rng, eta, n_max)))
    for n in (60, 120, 240, 480, 960):
        arms.append((f"fixed/n={n}/a=0.05", lambda rng, n=n: fixed_n(table, decs, 0.05, rng, n)))
    res = {}
    for name, f in arms:
        rng = np.random.default_rng(seed)
        c, w, r, rev, anyw = [], [], [], [], 0
        for _ in range(reps):
            cost, out = f(rng)
            wr, rt = score(out, tru)
            c.append(cost)
            w.append(wr)
            r.append(rt)
            rev.append(reversals(out, decs, tru))
            anyw += wr > 0
        res[name] = {"mean_runs": float(np.mean(c)), "p_any_wrong": anyw / reps, "mean_wrong": float(np.mean(w)),
                     "mean_correct_signs": float(np.mean(r)), "mean_reversals": float(np.mean(rev))}
        print(name, res[name], file=sys.stderr, flush=True)
    return {"K": len(decs), "reps": reps, "n_max": n_max, "truth": {"|".join(d): tru[d] for d in decs},
            "units": {m: len(us) for m, us in table.items()}, "arms": res}


def certify(table, alpha=0.05):
    """One-shot certification of every channel contrast on the full pools: R1 (RC-mix on delta) and R1'
    (plug-in on discordant pairs), one family of size K. Returns rows for the report."""
    decs = decisions_for(table)
    k = len(decs)
    rows = []
    for d in decs:
        us = list(table[d[0]].values())
        c, sg = ContrastCS(alpha / k), SignCS(alpha / k, cls=BettingCS)
        for u in us:
            c.update(u[d[1]], u[d[2]])
            sg.update(u[d[1]], u[d[2]])
        lo, hi = c.interval()
        slo, shi = sg.interval()
        rows.append({"model": d[0], "a": d[1], "b": d[2], "n": len(us), "discordant": sum(u[d[1]] != u[d[2]] for u in us),
                     "p_a": float(np.mean([u[d[1]] for u in us])), "p_b": float(np.mean([u[d[2]] for u in us])),
                     "delta": truth(table, d), "r1_cs": [lo, hi], "r1": sign_decision(lo, hi),
                     "r1p_cs": [slo, shi], "r1p": sign_decision(slo, shi)})
    return {"K": k, "alpha": alpha, "rows": rows,
            "reversals_r1p": reversals({d: r["r1p"] for d, r in zip(decs, rows)}, decs),
            "reversals_r1": reversals({d: r["r1"] for d, r in zip(decs, rows)}, decs)}


def _wilson(k, n, z):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def peek_select(table, rng, setting="null", looks=(1, 5, 20), n_rounds=500, alpha=0.05, eta=0.1):
    """E3' (plan.md, preregistered 52206bd5): one replay. Units are drawn i.i.d. per model per round, and every
    contrast is updated every round (open-strata blocks). setting 'null' permutes the 3 channel outcomes within each
    drawn unit (delta = 0 for every contrast). At each look the analyst takes each method's most extreme contrast and
    reports its sign if significant at that look. Returns {L: {method: erred}} plus the SPRT in-zone flag.

    Naive tests use Bonferroni alpha/K like the CS methods, so only peeking + selection separates them:
    McNemar = exact two-sided binomial test on discordant counts; Wilson = Wilson score interval for
    q = P(up | discordant) excluding 1/2 (the paired-difference sign)."""
    from scipy.stats import norm
    decs = decisions_for(table)
    k = len(decs)
    a = alpha / k
    z = norm.isf(a / 2)
    units = {m: list(us.values()) for m, us in table.items()}
    if setting == "null":
        tru = {d: 0.0 for d in decs}
    else:
        tru = {d: truth(table, d) for d in decs}
    # true q per contrast (for the SPRT zone split): q = P(up | discordant)
    qtrue = {}
    for d in decs:
        if setting == "null":
            qtrue[d] = 0.5
        else:
            us = units[d[0]]
            up = sum(u[d[1]] > u[d[2]] for u in us)
            dn = sum(u[d[1]] < u[d[2]] for u in us)
            qtrue[d] = up / (up + dn) if up + dn else 0.5
    r1 = {d: ContrastCS(a) for d in decs}
    r1p = {d: SignCS(a, cls=BettingCS) for d in decs}
    sp = {al: {d: [0.0, None] for d in decs} for al in (0.05, 0.01)}
    sp_b = {al: (math.log((1 - al / k) / (al / k)), math.log((al / k) / (1 - al / k))) for al in (0.05, 0.01)}
    up = {d: 0 for d in decs}
    dn = {d: 0 for d in decs}
    look_sets = {L: set(int(x) for x in rng.integers(20, n_rounds + 1, size=L)) for L in looks}
    all_looks = set().union(*look_sets.values())
    err = {L: collections.defaultdict(bool) for L in looks}
    sprt_zone = {L: collections.defaultdict(bool) for L in looks}
    chans = ("tool_return", "config", "tool_desc")
    for t in range(1, n_rounds + 1):
        draw = {}
        for m, us in units.items():
            u = us[rng.integers(len(us))]
            if setting == "null":
                v = [u[c] for c in chans]
                rng.shuffle(v)
                u = dict(zip(chans, v))
            draw[m] = u
        for d in decs:
            u = draw[d[0]]
            xa, xb = u[d[1]], u[d[2]]
            r1[d].update(xa, xb)
            r1p[d].update(xa, xb)
            up[d] += xa > xb
            dn[d] += xa < xb
            for al in sp:
                st = sp[al][d]
                if st[1] is None:
                    st[0] = paired_sprt_step(st[0], xa, xb, eta)
                    st[1] = ">" if st[0] >= sp_b[al][0] else "<" if st[0] <= sp_b[al][1] else None
        if t not in all_looks:
            continue
        rep = {}
        # CS methods: most extreme = interval farthest from 0
        for name, cs in (("R1", r1), ("R1prime", r1p)):
            best, bscore, bsign = None, -9, None
            for d in decs:
                lo, hi = cs[d].interval()
                sc = max(lo, -hi)
                if sc > bscore:
                    best, bscore, bsign = d, sc, sign_decision(lo, hi)
            rep[name] = (best, bsign)
        for al in sp:  # SPRT: among contrasts already decided, the one with largest |LLR|
            cand = [(abs(sp[al][d][0]), d) for d in decs if sp[al][d][1]]
            if cand:
                d = max(cand)[1]
                rep[f"SPRT{al}"] = (d, sp[al][d][1])
            else:
                rep[f"SPRT{al}"] = (None, None)
        best, bp = None, 2.0  # McNemar: smallest p-value
        for d in decs:
            n_ = up[d] + dn[d]
            p = binomtest(up[d], n_, 0.5).pvalue if n_ else 1.0
            if p < bp:
                best, bp = d, p
        rep["McNemar_peek"] = (best, (">" if up[best] > dn[best] else "<") if best and bp <= a else None)
        best, bsc, bs = None, -9, None  # Wilson on q
        for d in decs:
            lo, hi = _wilson(up[d], up[d] + dn[d], z)
            sc = max(lo - 0.5, 0.5 - hi)
            if sc > bsc:
                best, bsc, bs = d, sc, (">" if lo > 0.5 else "<" if hi < 0.5 else None)
        rep["Wilson_peek"] = (best, bs)
        for L, ls in look_sets.items():
            if t not in ls:
                continue
            for name, (d, sgn) in rep.items():
                if sgn is None:
                    continue
                wrong = tru[d] == 0 or (sgn == ">") != (tru[d] > 0)
                if wrong:
                    err[L][name] = True
                    if name.startswith("SPRT") and abs(qtrue[d] - 0.5) < eta:
                        sprt_zone[L][name] = True
    return err, sprt_zone


PEEK_METHODS = ("R1", "R1prime", "SPRT0.05", "SPRT0.01", "McNemar_peek", "Wilson_peek")


def peek_experiment(table, reps=400, seed=0, n_rounds=500):
    out = {}
    for setting in ("null", "real"):
        rng = np.random.default_rng(seed)
        cnt = {L: collections.Counter() for L in (1, 5, 20)}
        zone = {L: collections.Counter() for L in (1, 5, 20)}
        for i in range(reps):
            err, sz = peek_select(table, rng, setting, n_rounds=n_rounds)
            for L in cnt:
                for mth in PEEK_METHODS:
                    cnt[L][mth] += err[L][mth]
                    zone[L][mth] += sz[L][mth]
            if (i + 1) % 50 == 0:
                print(setting, i + 1, {L: dict(cnt[L]) for L in cnt}, file=sys.stderr, flush=True)
        out[setting] = {str(L): {mth: {"fwer": cnt[L][mth] / reps, "sprt_in_zone_errors": zone[L][mth] / reps}
                                 for mth in PEEK_METHODS} for L in cnt}
    return {"K": len(decisions_for(table)), "reps": reps, "n_rounds": n_rounds, "results": out}


BUDGETS = (900, 1800, 3600, 7200, 14400)


def budget_curve(table, reps=50, seed=0, n_max=2000):
    """Correct and wrong certified signs available at each budget B (anytime read-out). Sequential methods are
    read at B from their closure costs; fixed-n is only valid at its own n, so it contributes points, not a curve."""
    decs = decisions_for(table)
    tru = {d: truth(table, d) for d in decs}
    arms = {"riskcert/open_strata+tie0.05/a=0.05": lambda rng: riskcert(table, decs, 0.05, rng, "open_strata", eps=0.05, n_max=n_max),
            "riskcert/open_strata/a=0.05": lambda rng: riskcert(table, decs, 0.05, rng, "open_strata", n_max=n_max),
            "riskcert_sign/open_strata/a=0.05": lambda rng: riskcert(table, decs, 0.05, rng, "open_strata", n_max=n_max, cs_cls=BettingCS, stat="sign"),
            "riskcert_sign_mix/open_strata/a=0.05": lambda rng: riskcert(table, decs, 0.05, rng, "open_strata", n_max=n_max, cs_cls=RCMixCS, stat="sign"),
            "sprt_rr/eta=0.1/a=0.01": lambda rng: paired_sprt_rr(table, decs, 0.01, rng, 0.1, n_max),
            "sprt_rr/eta=0.1/a=0.05": lambda rng: paired_sprt_rr(table, decs, 0.05, rng, 0.1, n_max)}
    res = {}
    for name, f in arms.items():
        rng = np.random.default_rng(seed)
        fn = riskcert if name.startswith("riskcert") else paired_sprt_rr
        right = {b: [] for b in BUDGETS}
        wrong = {b: [] for b in BUDGETS}
        for _ in range(reps):
            _, out = f(rng)
            for b in BUDGETS:
                sub = {d: s for d, s in out.items() if fn.closed_at.get(d, 1e18) <= b}
                w, r = score(sub, tru)
                right[b].append(r)
                wrong[b].append(w)
        res[name] = {str(b): {"correct": float(np.mean(right[b])), "p_any_wrong": float(np.mean([x > 0 for x in wrong[b]]))}
                     for b in BUDGETS}
        print(name, res[name], file=sys.stderr, flush=True)
    return {"K": len(decs), "reps": reps, "budgets": BUDGETS, "arms": res}


if __name__ == "__main__":
    import rc_analyze
    stage = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    rows = rc_analyze.load("m0.jsonl", "pilot.jsonl", "pilot_llama31-8b-local.jsonl")
    if stage == "e1":  # pilot + E1 AgentDojo rows, one-shot certification
        rows += rc_analyze.load("e1_qwen3.jsonl", "e1_controls.jsonl", "e1_qwen25.jsonl")
        res = certify(pools_from_rows(rows))
        json.dump(res, open(os.path.join(HERE, "..", "results", "e1_agentdojo_certify.json"), "w"), indent=1)
        for r in res["rows"]:
            print(f"{r['model'][:20]:20s} {r['a'][:6]}-{r['b'][:6]} n={r['n']} disc={r['discordant']:3d} "
                  f"p={r['p_a']:.2f}/{r['p_b']:.2f} d={r['delta']:+.3f} R1={r['r1'] or '?'} R1'={r['r1p'] or '?'}")
        print("K", res["K"], "reversals R1'", res["reversals_r1p"], "R1", res["reversals_r1"])
    elif len(sys.argv) > 3 and sys.argv[3] == "budget":
        res = budget_curve(pools_from_rows(rows), reps=reps)
        json.dump(res, open(os.path.join(HERE, "..", "results", f"{stage}_budget_curve.json"), "w"), indent=1)
    else:
        res = sweep(pools_from_rows(rows), reps=reps)
        json.dump(res, open(os.path.join(HERE, "..", "results", f"{stage}_contrast_replay.json"), "w"), indent=1)
