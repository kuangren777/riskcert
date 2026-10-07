import json

import rc_analyze


def _row(key, model, ch, pos, sec, err=None):
    return {"key": key, "model": model, "channel": ch, "pos": pos, "security": sec, "err": err,
            "utility": True, "served_root": "x"}


def test_load_dedups_and_drops_err(tmp_path, monkeypatch):
    p = tmp_path / "a.jsonl"
    rows = [_row("k1", "m", "config", "start", None, err="boom"), _row("k1", "m", "config", "start", True),
            _row("k1", "m", "config", "start", False), _row("k2", "m", "config", "end", False)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(rc_analyze, "DATA", str(tmp_path))
    out = rc_analyze.load("a.jsonl")
    assert [(r["key"], r["security"]) for r in out] == [("k1", True), ("k2", False)]


def test_wilson_known_value():
    lo, hi = rc_analyze.wilson(36, 60)
    assert abs(lo - 0.47) < 0.01 and abs(hi - 0.71) < 0.01  # matches D15 4o-mini tool_return


def test_table_and_replay_shapes():
    rows = [_row(f"{c}{p}{i}", "m", c, p, i % 3 == 0) for c in rc_analyze.CH for p in rc_analyze.POS
            for i in range(12)]
    t = rc_analyze.table(rows)
    assert "| m | 36/108=0.33 |" in t
    rep = rc_analyze.replay_cost(rows, reps=5)
    assert rep["cells"] == 9 and rep["riskcert_mean_draws"] > 0


def test_load_drops_qwen3_rows_without_thinking_kwarg(tmp_path, monkeypatch):
    p = tmp_path / "q.jsonl"
    a = _row("a", "qwen3-8b-local", "config", "start", True)
    b = dict(_row("b", "qwen3-8b-local", "config", "start", False),
             extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    c = _row("c", "gpt-4o-mini", "config", "start", False)
    p.write_text("\n".join(json.dumps(r) for r in (a, b, c)) + "\n")
    monkeypatch.setattr(rc_analyze, "DATA", str(tmp_path))
    assert [r["key"] for r in rc_analyze.load("q.jsonl")] == ["b", "c"]
