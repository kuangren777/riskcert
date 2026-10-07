"""Build a result-audit directory for the pilot from raw jsonl, independent of rc_analyze.py.

Claimed numbers are parsed from results/pilot_table.md (what gets reported); recomputation reads
data/*.jsonl directly. Per-cell metrics use indicator fields `cell.<model>|<channel>` that are the
row's security outcome inside the cell and None outside (audit.py drops None before aggregating).

    python3 code/audit_prep.py pilot  ->  results/audit_pilot/  then run audit.py on it
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SHORT = {"gpt-4o-mini-2024-07-18": "4o", "gpt-4.1-mini-2025-04-14": "41", "gpt-5.4-nano-2026-03-17": "nano",
         "qwen3-8b-local": "qwen3", "llama31-8b-local": "llama"}
CHANNELS = ("tool_return", "config", "tool_desc")


def load_rows():
    rows = []
    for f in ("m0.jsonl", "pilot.jsonl", "pilot_llama31-8b-local.jsonl"):
        for line in open(os.path.join(ROOT, "data", f)):
            r = json.loads(line)
            if r.get("err") is not None or r["model"] not in SHORT:
                continue
            if r["model"] == "qwen3-8b-local" and not r.get("extra_body"):
                continue  # M0 Qwen rows predate enable_thinking=false (plan.md §2)
            rows.append(r)
    return rows


def claimed_from_table(path):
    """channel-level means and pooled fractions as printed in pilot_table.md."""
    out = {}
    for line in open(path):
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cols or cols[0] not in SHORT:
            continue
        s = SHORT[cols[0]]
        if len(cols) == 5:  # model | tool_return | config | tool_desc | utility
            for ch, c in zip(CHANNELS, cols[1:4]):
                out[f"{s}|{ch}"] = float(c.split()[0])
                out[f"n|{s}|{ch}"] = int(re.search(r"n=(\d+)", c).group(1))
        elif len(cols) == 11:  # model | pooled | 9 cells
            k, n = map(int, cols[1].split("=")[0].split("/"))
            out[f"pooled|{s}"] = k / n
    return out


def main(stage):
    rows = load_rows()
    d = os.path.join(ROOT, "results", f"audit_{stage}")
    os.makedirs(os.path.join(d, "samples"), exist_ok=True)
    samples = []
    for r in rows:
        s = SHORT[r["model"]]
        rec = {"key": r["key"], "model": r["model"], "channel": r["channel"], "security": bool(r["security"]),
               "exposed": bool(r.get("exposed")), "served_root": r.get("served_root"), "cell": {}}
        for m in SHORT.values():
            rec["cell"][f"pooled|{m}"] = rec["security"] if m == s else None
            for ch in CHANNELS:
                hit = m == s and ch == r["channel"]
                rec["cell"][f"{m}|{ch}"] = rec["security"] if hit else None
                rec["cell"][f"n|{m}|{ch}"] = 1 if hit else None
        samples.append(rec)
    json.dump(samples, open(os.path.join(d, "samples", "rows.json"), "w"))
    claimed = claimed_from_table(os.path.join(ROOT, "results", f"{stage}_table.md"))
    json.dump({"cell": claimed}, open(os.path.join(d, "summary.json"), "w"), indent=1)
    metrics = {}
    for name in claimed:
        if name.startswith("n|"):
            metrics[name] = {"field": f"cell.{name}", "agg": "count", "claimed": f"cell.{name}"}
        else:  # table prints 2 decimals
            metrics[name] = {"field": f"cell.{name}", "agg": "frac_true", "claimed": f"cell.{name}",
                             "tol": 0.005 if not name.startswith("pooled") else 1e-9}
    cfg = {"metrics": metrics,
           "polarity": [{"name": "unexposed_never_violates", "sample_filter": {"exposed": False, "channel": "tool_return"},  # exposed = tool-output exposure only
                         "field": "security", "expect": False, "min_fraction": 1.0}],
           "provenance": {"request_count_field": "request_count", "served_id_field": "served_ids",
                          "relay_log": f"../../data/{stage}.log", "min_requests": 1}}
    json.dump(cfg, open(os.path.join(d, "audit_config.json"), "w"), indent=1)
    json.dump({"request_count": len(rows), "served_ids": sorted({r.get("served_root") for r in rows})},
              open(os.path.join(d, "provenance.json"), "w"), indent=1)
    print(d, len(rows), "rows,", len(metrics), "metrics")





def K(name):
    """audit.py splits field paths on '.', and model names contain dots"""
    return name.replace(".", "_")


def main_e1():
    """E1 audit dirs: results/audit_e1_ia (InjecAgent) and results/audit_e1_ad (AgentDojo pilot + E1 pools).
    Claimed numbers come from the analysis outputs (results/e1_ia.json, results/e1_agentdojo_certify.json);
    recomputation reads data/*.jsonl directly."""
    import glob
    sys.path.insert(0, HERE)
    import ia_run  # case table only, for the attacker tool name (no model calls)
    act = re.compile(r"Action:\s*([A-Za-z0-9_]+)")
    # ---- InjecAgent
    claimed = json.load(open(os.path.join(ROOT, "results", "e1_ia.json")))
    rows, seen = [], set()
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "ia_e1_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r.get("err") is None and r["key"] not in seen:
                seen.add(r["key"])
                rows.append(r)
    d = os.path.join(ROOT, "results", "audit_e1_ia")
    os.makedirs(os.path.join(d, "samples"), exist_ok=True)
    samples, metrics, summary = [], {}, {}
    cells = sorted(claimed["cells"])
    for r in rows:
        atk = ia_run.CASES[(r["attack"], r["setting"])][r["case"]]["Attacker Tools"][0]
        named = atk in act.findall(r.get("output") or "")
        c = f"{r['model']}|{r['attack']}|{r['setting']}"
        rec = {"key": r["key"], "succ": r["eval"] == "succ", "atk_named_in_actions": named, "cell": {}}
        for k in cells:
            hit = k == c
            rec["cell"][K(f"all|{k}")] = (r["eval"] == "succ") if hit else None
            rec["cell"][K(f"valid|{k}")] = (r["eval"] == "succ") if hit and r["eval"] != "invalid" else None
            rec["cell"][K(f"inv|{k}")] = (r["eval"] == "invalid") if hit else None
        samples.append(rec)
    for k in cells:
        st = claimed["cells"][k]
        for tag, field in (("all", "asr_all"), ("valid", "asr_valid"), ("inv", "invalid_rate")):
            if st[field] is None:
                continue
            summary[K(f"{tag}|{k}")] = st[field]
            metrics[K(f"{tag}|{k}")] = {"field": "cell." + K(f"{tag}|{k}"), "agg": "frac_true", "claimed": "cell." + K(f"{tag}|{k}"), "tol": 1e-9}
    json.dump(samples, open(os.path.join(d, "samples", "rows.json"), "w"))
    json.dump({"cell": summary}, open(os.path.join(d, "summary.json"), "w"))
    json.dump({"metrics": {k: dict(v, claimed=v["claimed"]) for k, v in metrics.items()},
               "polarity": [{"name": "attacker_tool_never_named_never_succ", "sample_filter": {"atk_named_in_actions": False},
                             "field": "succ", "expect": False, "min_fraction": 1.0}],
               "provenance": {"request_count_field": "request_count", "served_id_field": "served_ids",
                              "relay_log": "../../data/ia_e1_gpt-4o-mini-2024-07-18.log", "min_requests": 1}},
              open(os.path.join(d, "audit_config.json"), "w"), indent=1)
    json.dump({"request_count": len(rows), "served_ids": sorted({r["served_root"] for r in rows})},
              open(os.path.join(d, "provenance.json"), "w"))
    print(d, len(rows), "rows,", len(metrics), "metrics")
    # ---- AgentDojo (pilot + E1)
    cert = json.load(open(os.path.join(ROOT, "results", "e1_agentdojo_certify.json")))
    ad = []
    for f in ("m0.jsonl", "pilot.jsonl", "pilot_llama31-8b-local.jsonl", "e1_qwen3.jsonl", "e1_controls.jsonl", "e1_qwen25.jsonl"):
        p = os.path.join(ROOT, "data", f)
        if os.path.exists(p):
            for line in open(p):
                r = json.loads(line)
                if r.get("err") is None and not (r["model"] == "qwen3-8b-local" and not r.get("extra_body")):
                    ad.append(r)
    d = os.path.join(ROOT, "results", "audit_e1_ad")
    os.makedirs(os.path.join(d, "samples"), exist_ok=True)
    claims, metrics = {}, {}
    for row in cert["rows"]:
        for ch, key in ((row["a"], "p_a"), (row["b"], "p_b")):
            claims[K(f"{row['model']}|{ch}")] = row[key]
    samples = []
    for r in ad:
        rec = {"key": r["key"], "channel": r["channel"], "security": bool(r["security"]), "exposed": bool(r.get("exposed")), "cell": {}}
        for k in claims:
            rec["cell"][k] = bool(r["security"]) if K(f"{r['model']}|{r['channel']}") == k else None
        samples.append(rec)
    for k, v in claims.items():
        metrics[k] = {"field": "cell." + k, "agg": "frac_true", "claimed": "cell." + k, "tol": 1e-9}
    json.dump(samples, open(os.path.join(d, "samples", "rows.json"), "w"))
    json.dump({"cell": claims}, open(os.path.join(d, "summary.json"), "w"))
    json.dump({"metrics": metrics,
               "polarity": [{"name": "unexposed_tool_return_never_violates", "sample_filter": {"exposed": False, "channel": "tool_return"},
                             "field": "security", "expect": False, "min_fraction": 1.0}],
               "provenance": {"request_count_field": "request_count", "served_id_field": "served_ids",
                              "relay_log": "../../data/e1_controls.log", "min_requests": 1}},
              open(os.path.join(d, "audit_config.json"), "w"), indent=1)
    json.dump({"request_count": len(ad), "served_ids": sorted({r.get("served_root") for r in ad})},
              open(os.path.join(d, "provenance.json"), "w"))
    print(d, len(ad), "rows,", len(metrics), "metrics")


def main_c1():
    """C1 audit: claimed per-model channel rates from results/c1.json, recomputed from data/confirm_c1.jsonl."""
    claimed = json.load(open(os.path.join(ROOT, "results", "c1.json")))
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "confirm_c1.jsonl"))]
    rows = [r for r in rows if r.get("err") is None]
    d = os.path.join(ROOT, "results", "audit_c1")
    os.makedirs(os.path.join(d, "samples"), exist_ok=True)
    keys = {K(f"{m}|{ch}"): (m, ch) for m in claimed["models"] for ch in ("tool_return", "config")}
    samples = []
    for r in rows:
        rec = {"key": r["key"], "channel": r["channel"], "security": bool(r["security"]), "exposed": bool(r.get("exposed")),
               "no_tool_calls": not (r.get("trace") or []), "cell": {}}
        for k, (m, ch) in keys.items():
            rec["cell"][k] = bool(r["security"]) if (r["model"], r["channel"]) == (m, ch) else None
        samples.append(rec)
    summary = {k: claimed["models"][m][f"p_{ch}"] for k, (m, ch) in keys.items()}
    json.dump(samples, open(os.path.join(d, "samples", "rows.json"), "w"))
    json.dump({"cell": summary}, open(os.path.join(d, "summary.json"), "w"))
    json.dump({"metrics": {k: {"field": "cell." + k, "agg": "frac_true", "claimed": "cell." + k, "tol": 1e-9} for k in keys},
               # every C1 tool_return episode is exposed, so anchor on: no tool call -> env state unchanged -> no violation
               "polarity": [{"name": "no_tool_calls_never_violates", "sample_filter": {"no_tool_calls": True},
                             "field": "security", "expect": False, "min_fraction": 1.0}],
               "provenance": {"request_count_field": "request_count", "served_id_field": "served_ids",
                              "relay_log": "../../data/confirm_c1.log", "min_requests": 1}},
              open(os.path.join(d, "audit_config.json"), "w"), indent=1)
    json.dump({"request_count": len(rows), "served_ids": sorted({r["served_root"] for r in rows})}, open(os.path.join(d, "provenance.json"), "w"))
    print(d, len(rows), "rows")


def main_c1b():
    """C1b audit: per-model channel success counts from results/c1b.json, recomputed from data/confirm_c1b.jsonl
    (first err-free row per key, the primary analysis)."""
    claimed = json.load(open(os.path.join(ROOT, "results", "c1b.json")))
    seen, rows = set(), []
    for line in open(os.path.join(ROOT, "data", "confirm_c1b.jsonl")):
        r = json.loads(line)
        if r.get("err") is None and r["key"] not in seen:
            seen.add(r["key"])
            rows.append(r)
    nullp = os.path.join(ROOT, "data", "confirm_c1b_null.jsonl")  # null-agent oracle controls on the same grid
    controls = [json.loads(l) for l in open(nullp)] if os.path.exists(nullp) else []
    d = os.path.join(ROOT, "results", "audit_c1b")
    os.makedirs(os.path.join(d, "samples"), exist_ok=True)
    keys = {K(f"{m}|{ch}"): (m, ch) for m in claimed["primary"] for ch in ("tool_return", "config")}
    samples = []
    for r in rows:
        rec = {"key": r["key"], "security": bool(r["security"]), "no_tool_calls": not (r.get("trace") or []), "cell": {}}
        for k, (m, ch) in keys.items():
            rec["cell"][k] = bool(r["security"]) if (r["model"], r["channel"]) == (m, ch) else None
        samples.append(rec)
    for r in controls:  # anchor rows only: no cell values, so they never enter a recomputed metric
        samples.append({"key": r["key"], "security": bool(r["security"]), "no_tool_calls": not (r.get("trace") or []),
                        "cell": {k: None for k in keys}})
    summary = {k: claimed["primary"][m][ch] for k, (m, ch) in keys.items()}
    json.dump(samples, open(os.path.join(d, "samples", "rows.json"), "w"))
    json.dump({"cell": summary}, open(os.path.join(d, "summary.json"), "w"))
    json.dump({"metrics": {k: {"field": "cell." + k, "agg": "sum", "claimed": "cell." + k, "tol": 1e-9} for k in keys},
               "polarity": [{"name": "no_tool_calls_never_violates", "sample_filter": {"no_tool_calls": True},
                             "field": "security", "expect": False, "min_fraction": 1.0}],
               "provenance": {"request_count_field": "request_count", "served_id_field": "served_ids",
                              "relay_log": "../../data/confirm_c1b.log", "min_requests": 1}},
              open(os.path.join(d, "audit_config.json"), "w"), indent=1)
    json.dump({"request_count": len(rows), "served_ids": sorted({r["served_root"] for r in rows})}, open(os.path.join(d, "provenance.json"), "w"))
    print(d, len(rows), "rows; no-tool-call rows", sum(s["no_tool_calls"] for s in samples))


def main_sessb():
    """Session-B audit: per (model, channel) violation rates behind C2 (results/c2.json rows p_a / p_b),
    recomputed from data/e2b_*.jsonl (first err-free row per key, null-agent excluded). Anchor = null-agent rows."""
    import glob
    cert = json.load(open(os.path.join(ROOT, "results", "c2.json")))
    claims = {}
    for row in cert["rows"]:
        claims[K(f"{row['model']}|{row['a']}")] = row["p_a"]
        claims[K(f"{row['model']}|{row['b']}")] = row["p_b"]
    rows, seen, controls = [], set(), []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "e2b_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r.get("err") is not None or r["key"] in seen:
                continue
            seen.add(r["key"])
            (controls if r["model"] == "null-agent" else rows).append(r)
    d = os.path.join(ROOT, "results", "audit_sessb")
    os.makedirs(os.path.join(d, "samples"), exist_ok=True)
    samples = []
    for r in rows + controls:
        k = K(f"{r['model']}|{r['channel']}")
        samples.append({"key": r["key"], "security": bool(r["security"]), "no_tool_calls": not (r.get("trace") or []),
                        "cell": {c: (bool(r["security"]) if c == k and r["model"] != "null-agent" else None) for c in claims}})
    json.dump(samples, open(os.path.join(d, "samples", "rows.json"), "w"))
    json.dump({"cell": claims}, open(os.path.join(d, "summary.json"), "w"))
    json.dump({"metrics": {c: {"field": "cell." + c, "agg": "frac_true", "claimed": "cell." + c, "tol": 1e-9} for c in claims},
               "polarity": [{"name": "no_tool_calls_never_violates", "sample_filter": {"no_tool_calls": True},
                             "field": "security", "expect": False, "min_fraction": 1.0}],
               "provenance": {"request_count_field": "request_count", "served_id_field": "served_ids",
                              "relay_log": "../../data/e2b_hub.log", "min_requests": 1}},
              open(os.path.join(d, "audit_config.json"), "w"), indent=1)
    json.dump({"request_count": len(rows), "served_ids": sorted({str(r.get("served_root")) for r in rows})},
              open(os.path.join(d, "provenance.json"), "w"))
    print(d, len(rows), "rows;", len(controls), "null-agent controls;", len(claims), "metrics")


def main_m3b():
    """M3b audit. E4: mean cost per (transition, arm) recomputed from the per-replay cost vectors in
    results/e4_replay_*.json. E5: species at stop and new species in continuation recomputed per cell with an
    independent loop (stop draw taken from results/e5.json), violations recomputed from raw rows.
    Anchor: E5 episodes without any tool call never violate."""
    import glob
    sys.path.insert(0, HERE)
    import e5_run
    d = os.path.join(ROOT, "results", "audit_m3b")
    os.makedirs(os.path.join(d, "samples"), exist_ok=True)
    claims, samples, metrics = {}, [], {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "e4_replay_*.json"))):
        for t in json.load(open(f)):
            for arm, v in t["arms"].items():
                k = K(f"e4|{t['new']}|{arm}|mean_cost")
                claims[k] = v["mean_cost"]
                for c in v["cost"]:
                    samples.append({"no_tool_calls": False, "security": False, "cell": {k: c}})
                metrics[k] = {"field": "cell." + k, "agg": "mean", "claimed": "cell." + k, "tol": 1e-6}
    e5 = json.load(open(os.path.join(ROOT, "results", "e5.json")))["cells"]
    for i, c in enumerate(e5_run.CELLS):
        name = e5_run.cell_name(c)
        rows = {}
        for line in open(os.path.join(ROOT, "data", f"e5_{name}.jsonl")):
            r = json.loads(line)
            if r.get("err") is None:
                rows.setdefault(r["draw"], r)
        seq = [rows[j] for j in range(len(rows)) if j in rows]
        n0 = e5[name]["stop_n"]
        seen, new_cont = set(), 0
        for j, r in enumerate(seq):
            sp = None
            if r.get("security"):
                sp = (r["it"], tuple(t["f"] for t in (r.get("trace") or []) if str(t.get("f", "")).startswith(
                    ("send", "update", "delete", "create", "add", "invite", "remove", "post", "reserve", "cancel",
                     "schedule", "write", "share", "append"))))
            if j < n0:
                if sp is not None:
                    seen.add(sp)
            elif j < n0 + 300 and sp is not None and sp not in seen:
                new_cont += 1
                seen.add(sp)
        ks, kn = K(f"e5|{name}|species_at_stop"), K(f"e5|{name}|new_in_cont")
        claims[ks], claims[kn] = e5[name]["species_at_stop"], e5[name]["new_in_cont"]
        # independent recomputation enters as one sample per metric with the recomputed value
        samples.append({"no_tool_calls": False, "security": False, "cell": {ks: _species_at_stop(seq, n0), kn: new_cont}})
        metrics[ks] = {"field": "cell." + ks, "agg": "max", "claimed": "cell." + ks, "tol": 0}
        metrics[kn] = {"field": "cell." + kn, "agg": "max", "claimed": "cell." + kn, "tol": 0}
        for r in seq:
            samples.append({"no_tool_calls": not (r.get("trace") or []), "security": bool(r.get("security")), "cell": {}})
    json.dump(samples, open(os.path.join(d, "samples", "rows.json"), "w"))
    json.dump({"cell": claims}, open(os.path.join(d, "summary.json"), "w"))
    json.dump({"metrics": metrics,
               "polarity": [{"name": "no_tool_calls_never_violates", "sample_filter": {"no_tool_calls": True},
                             "field": "security", "expect": False, "min_fraction": 1.0}],
               "provenance": {"request_count_field": "request_count", "served_id_field": "served_ids",
                              "relay_log": "../../data/e5_cell0.log", "min_requests": 1}},
              open(os.path.join(d, "audit_config.json"), "w"), indent=1)
    roots = set()
    for f in glob.glob(os.path.join(ROOT, "data", "e5_*__*.jsonl")):
        for line in open(f):
            roots.add(str(json.loads(line).get("served_root")))
    json.dump({"request_count": sum(1 for s in samples if s["cell"] == {}), "served_ids": sorted(roots)},
              open(os.path.join(d, "provenance.json"), "w"))
    print(d, len(metrics), "metrics")


def main_c3():
    """C3 audit: per-model success counts over the used rounds (results/c3.json), recomputed from the raw rows.
    Also checks that every used row's unit and position match the frozen unit list and the seeded draw sequence."""
    import c3_analyze as A
    import c3_run as C
    claimed = json.load(open(os.path.join(ROOT, "results", f"{C.PREFIX}.json")))["models"]
    U = C.units()
    d = os.path.join(ROOT, "results", f"audit_{C.PREFIX}")
    os.makedirs(os.path.join(d, "samples"), exist_ok=True)
    samples, summary, roots, n_req = [], {}, set(), 0
    for idx, m in enumerate(C.MODELS):
        if m not in claimed:
            continue
        rows = A.load(m)
        n_req += len(rows)
        n = claimed[m]["rounds"]
        seq = C.draws(idx)
        used = {}
        for r in rows:
            if r.get("err") is None and r["round"] < n:
                used.setdefault((r["round"], r["channel"]), r)
        assert len(used) == 2 * n, (m, len(used), n)
        for (rk, ch), r in used.items():
            s_, u, i, pos = U[seq[rk]]
            assert r["unit"] == seq[rk] and (r["suite"], r["ut"], r["it"], r["pos"]) == (s_, u, i, pos), (m, rk, ch)
            roots.add(str(r.get("served_root")))
            k = K(f"{m}|{ch}")
            samples.append({"key": r["key"], "security": bool(r["security"]), "no_tool_calls": not (r.get("trace") or []),
                            "cell": {k: bool(r["security"])}})
        summary[K(f"{m}|tool_return")] = round(claimed[m]["asr_tool_return"][0] * n)
        summary[K(f"{m}|config")] = round(claimed[m]["asr_config"][0] * n)
    for line in open(os.path.join(ROOT, "data", "c3_null.jsonl")):  # null-agent anchor rows, no cell values
        r = json.loads(line)
        samples.append({"key": r["key"], "security": bool(r["security"]), "no_tool_calls": not (r.get("trace") or []), "cell": {}})
    keys = list(summary)
    for smp in samples:
        smp["cell"] = {k: smp["cell"].get(k) for k in keys}
    json.dump(samples, open(os.path.join(d, "samples", "rows.json"), "w"))
    json.dump({"cell": summary}, open(os.path.join(d, "summary.json"), "w"))
    json.dump({"metrics": {k: {"field": "cell." + k, "agg": "sum", "claimed": "cell." + k, "tol": 1e-9} for k in keys},
               "polarity": [{"name": "no_tool_calls_never_violates", "sample_filter": {"no_tool_calls": True},
                             "field": "security", "expect": False, "min_fraction": 1.0}],
               "provenance": {"request_count_field": "request_count", "served_id_field": "served_ids",
                              "relay_log": f"../../data/{C.PREFIX}_qwen3-8b-local.log", "min_requests": 1}},
              open(os.path.join(d, "audit_config.json"), "w"), indent=1)
    json.dump({"request_count": n_req, "served_ids": sorted(roots)}, open(os.path.join(d, "provenance.json"), "w"))
    print(d, len(samples), "samples; no-tool-call rows", sum(s["no_tool_calls"] for s in samples))


def _species_at_stop(seq, n0):
    out = set()
    for r in seq[:n0]:
        if r.get("security"):
            out.add((r["it"], tuple(t["f"] for t in (r.get("trace") or []) if str(t.get("f", "")).startswith(
                ("send", "update", "delete", "create", "add", "invite", "remove", "post", "reserve", "cancel",
                 "schedule", "write", "share", "append")))))
    return len(out)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    {"e1": main_e1, "c1": main_c1, "c1b": main_c1b, "sessb": main_sessb, "m3b": main_m3b, "c3": main_c3}.get(stage, lambda: main(stage))()
