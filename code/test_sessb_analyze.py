"""Synthetic-data tests for sessb_analyze (never touch real session-B outcomes)."""
import json

import numpy as np

import sessb_analyze as S

CH = ("tool_return", "config", "tool_desc")
POS = ("start", "middle", "end")


def _ad_rows(ps, n_units=20, reps=(0, 1), seed=0, model_prefix="m"):
    rng = np.random.default_rng(seed)
    rows = []
    for j, p in enumerate(ps):
        m = f"{model_prefix}{j}"
        for u in range(n_units):
            for pos in POS:
                for r in reps:
                    for ch, pc in zip(CH, p):
                        rows.append({"model": m, "channel": ch, "pos": pos, "security": bool(rng.random() < pc),
                                     "key": f"{m}|{ch}|{pos}|s|u{u}|i|{r}", "err": None})
    return rows


def _ia_rows(p_succ=0.3, p_inv=0.1, n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n):
        for a in ("dh", "ds"):
            for s in ("base", "enhanced"):
                x = rng.random()
                ev = "invalid" if x < p_inv else ("succ" if x < p_inv + p_succ else "unsucc")
                rows.append({"model": "m", "attack": a, "setting": s, "eval": ev, "security": ev == "succ",
                             "key": f"m|ia|{a}|{s}|{c}|0", "err": None})
    return rows


def test_loader_excludes_null_and_err(tmp_path):
    rows = [{"model": "null-agent", "key": "a", "err": None, "security": False},
            {"model": "m", "key": "b", "err": "x", "security": None},
            {"model": "m", "key": "c", "err": None, "security": True},
            {"model": "m", "key": "c", "err": None, "security": False}]
    p = tmp_path / "e2b_x.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    out = S.load_ad("B", data=str(tmp_path))
    assert [r["key"] for r in out] == ["c"] and out[0]["security"] is True


def test_c2_runs_and_counts_on_synthetic():
    b = _ad_rows([(0.6, 0.1, 0.3), (0.05, 0.4, 0.2)], n_units=20, reps=(10, 11, 12, 13))
    a = _ad_rows([(0.6, 0.1, 0.3), (0.05, 0.4, 0.2)], n_units=20, reps=(0, 1), seed=1)
    res, md = S.c2(b, a, budget_reps=0)
    assert res["K"] == 6 and res["certified_r1prime"] >= res["certified_r1"]
    assert res["agree_with_A"] + len(res["disagree_with_A"]) == res["certified_r1prime"]
    assert "C2" in md[0]


def test_within_session_valid_methods_cover_and_wilson_peeking_worse():
    cells = {f"c{i}": list((np.random.default_rng(i).random(60) < 0.3).astype(int)) for i in range(6)}
    w, k = S.within_session(cells, alpha=0.2, reps=60)
    assert k == 6
    assert w["RC-mix"]["simultaneous_miss"] <= 0.2 + 3 * np.sqrt(0.2 * 0.8 / 60)
    assert w["plug-in"]["simultaneous_miss"] <= 0.2 + 3 * np.sqrt(0.2 * 0.8 / 60)
    assert w["Wilson-peek"]["marginal_miss"] >= w["RC-mix"]["marginal_miss"]


def test_cross_session_expected_miss_small_when_same_law():
    rng = np.random.default_rng(3)
    ca = {f"c{i}": list((rng.random(60) < 0.3).astype(int)) for i in range(5)}
    cb = {f"c{i}": list((rng.random(120) < 0.3).astype(int)) for i in range(5)}
    x = S.cross_session(ca, cb, boot=500)
    assert 0 <= x["RC-mix"]["expected_miss_from_B_noise"] <= 0.2 and x["RC-mix"]["cells"] == 5


def test_ia_cells_two_metrics():
    rows = _ia_rows()
    v, a = S.cells_ia(rows), S.cells_ia(rows, "asr_all")
    assert sum(len(x) for x in a.values()) == len(rows) > sum(len(x) for x in v.values())


def test_figures_from_synthetic_results(tmp_path):
    import make_figs, rc_replay
    L = {str(l): {m: {"fwer": 0.01 * i, "sprt_in_zone_errors": 0.0} for i, m in enumerate(rc_replay.PEEK_METHODS)} for l in (1, 5, 20)}
    json.dump({"results": {"null": L, "real": L}}, open(tmp_path / "e3p.json", "w"))
    w = {m: {"marginal_miss": 0.01, "simultaneous_miss": 0.03, "mean_final_width": 0.2} for m in ("RC-mix", "plug-in", "Wilson-peek", "CP-final")}
    json.dump({"AgentDojo": {"within_session_A": w}, "InjecAgent ASR-valid": {"within_session_A": w}}, open(tmp_path / "rq2.json", "w"))
    json.dump({"K": 3, "certified_r1": 1, "certified_r1prime": 2, "rows": [{"model": "glm-5.3", "a": "tool_return", "b": "config", "p_a": 0.0, "p_b": 0.3}, {"model": "glm-5.3", "a": "tool_return", "b": "tool_desc", "p_a": 0.0, "p_b": 0.1}]}, open(tmp_path / "c2.json", "w"))
    done = make_figs.main(str(tmp_path), str(tmp_path / "figs"))
    assert set(done) == {"e3p", "c2", "rq1"} and all((tmp_path / "figs" / f"fig_{n}.pdf").stat().st_size > 1000 for n in done)
