import json
import pytest

import rc_run


def test_grid_shape_and_unique_keys():
    jobs = rc_run.jobs_for(["m"], reps=2)
    assert len(jobs) == 15 * 3 * 3 * 2
    assert len({j["key"] for j in jobs}) == len(jobs)
    assert {j["channel"] for j in jobs} == set(rc_run.CHANNELS)
    assert {j["meta"]["pos"] for j in jobs} == {"start", "middle", "end"}


def test_grid_pairs_match_n03():
    assert rc_run.grid_pairs() == rc_run.grid_pairs()
    assert len(set(rc_run.grid_pairs())) == 15


def test_payload_positions_contain_goal_once():
    s, _, i = rc_run.grid_pairs()[0]
    goal = rc_run.harness.suite(s).injection_tasks[i].GOAL
    p = rc_run.positions(s, i)
    assert all(t.count(goal) == 1 for t in p.values())
    assert p["start"].index(goal) < p["middle"].index(goal) < p["end"].index(goal)


def test_root_mismatch_aborts(monkeypatch):
    monkeypatch.setattr(rc_run, "served_root", lambda m: "/models/Some-Other-Model")
    with pytest.raises(SystemExit, match="ABORT"):
        rc_run.check_roots(["qwen3-8b-local"])


def test_api_models_skip_root_check(monkeypatch):
    monkeypatch.setattr(rc_run, "served_root", lambda m: pytest.fail("should not query"))
    rc_run.check_roots(["gpt-4o-mini-2024-07-18"])
    assert rc_run._ROOT["gpt-4o-mini-2024-07-18"].startswith("api-gateway:")


def test_harness_routed_through_local_map():
    assert rc_run.harness.client_for is rc_run.client_for
    assert "qwen25-7b-local" in rc_run.harness.LOCAL_MODELS


def test_qwen3_client_injects_enable_thinking_false():
    c = rc_run.client_for("qwen3-8b-local")
    seen = {}
    c.chat.completions._inner = type("I", (), {"create": lambda self, **kw: seen.update(kw)})()
    c.chat.completions.create(model="qwen3-8b-local", messages=[], extra_body={"x": 1})
    assert seen["extra_body"] == {"x": 1, "chat_template_kwargs": {"enable_thinking": False}}
    rc_run._CLIENTS.pop("qwen3-8b-local")


def test_api_client_is_plain():
    assert rc_run.client_for("gpt-4o-mini-2024-07-18") is rc_run.harness.CLIENT


def test_run_jobs_caps_concurrency_and_records_extra_body(tmp_path, monkeypatch):
    import threading, time
    live, peak, lock = [0], [0], threading.Lock()

    def fake_run_pair(**kw):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.01)
        with lock:
            live[0] -= 1
        return {"model": kw["model"], "err": None, "security": False}

    monkeypatch.setattr(rc_run.harness, "run_pair", fake_run_pair)
    monkeypatch.setattr(rc_run, "check_roots", lambda ms: rc_run._ROOT.update({m: "r" for m in ms}))
    monkeypatch.setattr(rc_run, "_metrics_ok", lambda port: False)  # server busy -> base cap
    jobs = rc_run.jobs_for(["qwen3-8b-local"], reps=1)[:40]
    out = tmp_path / "o.jsonl"
    rc_run.run_jobs(jobs, str(out), threads=32)
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 40 and peak[0] <= rc_run.MAX_CONC
    assert rows[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_llama_client_uses_harness_single_call_wrapper():
    c = rc_run.client_for("llama31-8b-local")
    assert isinstance(c._client, rc_run.harness._SingleCallClient)
    assert rc_run.MSG_TRANSFORM_NAME["llama31-8b-local"] == "harness._SingleCallClient"
    rc_run._CLIENTS.pop("llama31-8b-local")


def test_llama_request_gets_split_and_system_role():
    seen = {}
    tc = lambda i: {"id": f"c{i}", "type": "function", "function": {"name": f"f{i}", "arguments": "{}"}}
    msg = lambda k: type("M", (), {"tool_calls": [tc(n) for n in range(k)]})()
    resp = lambda k: type("R", (), {"choices": [type("C", (), {"message": msg(k)})()]})()
    inner = type("O", (), {})()
    inner.chat = type("Ch", (), {})()
    inner.chat.completions = type("I", (), {"create": lambda self, **kw: (seen.update(kw), resp(3))[1]})()
    c = rc_run._ExtraBodyClient(rc_run.harness._SingleCallClient(inner), None)
    msgs = [{"role": "developer", "content": "sys"},
            {"role": "assistant", "content": "x", "tool_calls": [tc(1), tc(2)]},
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "content": "r2"}]
    rc_run._reset_episode()
    c.chat.completions.create(model="llama31-8b-local", messages=msgs)
    assert [m["role"] for m in seen["messages"]] == ["system", "assistant", "tool", "assistant", "tool"]
    assert rc_run.episode_calls() == {"model_turns": 1, "multi_call_turns": 1, "max_calls_per_turn": 3}


def test_local_b_ports_and_backend():
    assert rc_run.LOCAL["qwen25-7b-local"][0] == 8033 and rc_run.LOCAL["qwen3-32b-local"][0] == 8031
    assert rc_run.backend_of("qwen25-7b-local") == "local_b" and rc_run.backend_of("qwen3-8b-local") == "local_a"
    assert rc_run.backend_of("gpt-4o-mini-2024-07-18") == "api-gateway"


def test_gate_bursts_only_when_metrics_ok(monkeypatch):
    # c3_run lowers the qwen3 burst at import (C3 cap 8); set it explicitly so the result does not depend on test order
    monkeypatch.setitem(rc_run.BURST, "qwen3-8b-local", 16)
    g = rc_run.Gate("qwen3-8b-local")
    monkeypatch.setattr(rc_run, "_metrics_ok", lambda port: True)
    assert g.cap() == 16
    g._t = 0.0
    monkeypatch.setattr(rc_run, "_metrics_ok", lambda port: False)
    assert g.cap() == rc_run.MAX_CONC
    assert rc_run.Gate("llama31-8b-local").cap() == rc_run.MAX_CONC  # 8023 stays at 8
    assert rc_run.Gate("gpt-4o-mini-2024-07-18").cap() == rc_run.MAX_CONC


def test_metrics_parser(monkeypatch):
    import io
    body = ('vllm:num_requests_waiting{engine="0"} 0.0\nvllm:gpu_cache_usage_perc{engine="0"} 0.31\n')
    monkeypatch.setattr(rc_run.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body.encode()))
    assert rc_run._metrics_ok(8021)
    body = ('vllm:num_requests_waiting{engine="0"} 2.0\nvllm:gpu_cache_usage_perc{engine="0"} 0.31\n')
    assert not rc_run._metrics_ok(8021)


def test_confirm_pairs_disjoint_and_fixed():
    c = rc_run.confirm_pairs()
    assert len(c) == 15 and not set(c) & set(rc_run.grid_pairs()) and c == rc_run.confirm_pairs()
    jobs = rc_run.jobs_for(["m"], reps=4, pair_set=c, channels=("tool_return", "config"))
    assert len(jobs) == 15 * 3 * 2 * 4


def test_null_agent_episode_never_calls_tools_and_never_violates():
    rc_run.check_roots([rc_run.NULL_AGENT])
    j = rc_run.jobs_for([rc_run.NULL_AGENT], reps=1)[0]
    kw = {k: v for k, v in j.items() if k not in ("key", "meta")}
    r = rc_run.harness.run_pair(**kw)
    assert r["err"] is None, r["err"]
    assert not r["trace"] and r["security"] is False
    assert rc_run.backend_of(rc_run.NULL_AGENT) == "stub"


def test_metrics_parser_vllm_019_names_and_missing(monkeypatch):
    import io
    bodies = {"new_ok": 'vllm:num_requests_waiting{engine="0"} 0.0\nvllm:kv_cache_usage_perc{engine="0"} 0.5\n',
              "new_hot": 'vllm:num_requests_waiting{engine="0"} 0.0\nvllm:kv_cache_usage_perc{engine="0"} 0.89\n',
              "no_kv": 'vllm:num_requests_waiting{engine="0"} 0.0\n',
              "no_waiting": 'vllm:kv_cache_usage_perc{engine="0"} 0.1\n'}
    for name, want in (("new_ok", True), ("new_hot", False), ("no_kv", False), ("no_waiting", False)):
        monkeypatch.setattr(rc_run.urllib.request, "urlopen", lambda *a, b=bodies[name], **k: io.BytesIO(b.encode()))
        assert rc_run._metrics_ok(8031) is want, name


def test_confirm_b_pairs_new_goals_fixed():
    c = rc_run.confirm_b_pairs()
    used = {(s, i) for s, _, i in rc_run.grid_pairs() + rc_run.confirm_pairs()}
    assert len(c) == 18 and len(set(c)) == 18 and c == rc_run.confirm_b_pairs()
    assert not {(s, i) for s, _, i in c} & used
    assert len(rc_run.jobs_for(["m"], reps=4, pair_set=c, channels=("tool_return", "config"))) == 432
