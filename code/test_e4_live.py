"""Synthetic test for e4_live: run_pair is stubbed, nothing calls a model."""
import json
import random

import e4_live as L


def test_live_arm_with_stubbed_runs(tmp_path, monkeypatch):
    rnd = random.Random(0)
    p = {"tool_return": 0.5, "config": 0.02, "tool_desc": 0.2}
    monkeypatch.setattr(L.rc_run.harness, "run_pair",
                        lambda **kw: {"err": None, "security": rnd.random() < p[kw["channel"]], "trace": []})
    monkeypatch.setattr(L.rc_run, "check_roots", lambda ms: L.rc_run._ROOT.update({m: "stub" for m in ms}))
    monkeypatch.setattr(L.S, "DATA", str(tmp_path))
    res = L.run_live("gpt-5.4-nano-2026-03-17", "gpt-4.1-mini-2025-04-14", "cold")
    assert res["cost"] <= 540 and res["rounds"] <= L.ROUNDS
    used = json.load(open(tmp_path / "e4_live_gpt-5.4-nano-2026-03-17_cold.used.json"))["used"]
    assert len(used) == res["cost"]
