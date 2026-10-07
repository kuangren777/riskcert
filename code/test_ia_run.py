import os

import pytest

_IA = os.environ.get("INJECAGENT_DIR", os.path.join(os.path.dirname(__file__), "..", "third_party", "InjecAgent"))
if not os.path.isdir(_IA):
    pytest.skip("InjecAgent checkout not found (set INJECAGENT_DIR)", allow_module_level=True)
import ia_run as I  # noqa: E402


def test_cases_aligned_and_units_paired():
    for a in I.ATTACKS:
        b, e = I.CASES[(a, "base")], I.CASES[(a, "enhanced")]
        assert len(b) == len(e)
        assert all(x["Attacker Instruction"] == y["Attacker Instruction"] for x, y in zip(b, e))
    jobs = I.jobs_for(["m"], n_units=5, reps=2)
    assert len(jobs) == 2 * 5 * 2 * 2 and len({j["key"] for j in jobs}) == len(jobs)
    assert I.units("dh", 5) == I.units("dh", 5)


def test_prompt_contains_injection_and_tools():
    item = I.CASES[("dh", "enhanced")][0]
    msgs = I.prompt(item)
    assert msgs[0]["role"] == "system" and "IMPORTANT!!!" in msgs[1]["content"]
    assert item["Attacker Tools"][0] in msgs[1]["content"]


def test_parser_polarity_on_synthetic_outputs():
    item = I.CASES[("dh", "base")][0]
    atk, user = item["Attacker Tools"][0], item["User Tool"]
    succ = f"Thought: I will do it.\nAction: {atk}\nAction Input: {{}}\n"
    safe = "Thought: The product details are retrieved.\nFinal Answer: Here are the details of the laptop."
    ev = lambda t: I.PARSE.evaluate_output_prompted(t, atk, item["Attacker Tools"], user, item["Attacker Instruction"])["eval"]
    assert ev(succ) == "succ" and ev(safe) == "unsucc"


def test_run_one_records_error_without_raising(monkeypatch):
    def boom(m):
        raise RuntimeError("down")
    monkeypatch.setattr(I.rc_run, "client_for", boom)
    I.rc_run._ROOT["m"] = "r"
    j = I.jobs_for(["m"], n_units=1, reps=1)[0]
    r = I.run_one(j)
    assert r["security"] is None and "down" in r["err"]
