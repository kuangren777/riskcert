"""RiskCert runner for InjecAgent (second benchmark), reusing ISSTA26/InjecAgent read-only.

Strata: attack {dh, ds} x setting {base, enhanced}; attack type (3 per attack) is a case property used as a
covariate. Unit = (attack, case index): base and enhanced files are aligned case by case, so one unit runs
under both settings (paired). Prompt = InjecAgent's own "InjecAgent" ReAct prompt; oracle = its deterministic
parser `evaluate_output_prompted` (violation = eval 'succ', i.e. the attacker tool is called). ds is scored on
step 1 only (attacker tool called), which avoids InjecAgent's gpt-4 simulated step-2 responses.
Free-form outputs that the parser marks 'invalid' count as no violation; the invalid rate is reported.

Sampling: T=1 (Bernoulli estimand, same as the AgentDojo runs; InjecAgent's default is T=0), max_tokens=1024.
Local models go through rc_run.client_for (port map, root check, Qwen3 enable_thinking=false).

    python3 ia_run.py m0        # 3 models x 2 attacks x 2 settings x 10 units x 1 rep = 120 calls
"""
import importlib.util
import json
import os
import random
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
IA = os.path.abspath(os.environ.get("INJECAGENT_DIR", os.path.join(HERE, "..", "third_party", "InjecAgent")))
sys.path.insert(0, HERE)
import rc_run  # noqa: E402  (port map, root check, clients)


def _ngrams(seq, n):  # stand-in for nltk.ngrams (only symbol output_parsing uses), same semantics
    seq = list(seq)
    return zip(*(seq[i:] for i in range(n)))


sys.modules.setdefault("nltk", types.SimpleNamespace(ngrams=_ngrams))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(IA, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROMPTS = _load("ia_agent_prompts", "src/prompts/agent_prompts.py")
PARSE = _load("ia_output_parsing", "src/output_parsing.py")
SYS_PROMPT, USER_PROMPT = PROMPTS.PROMPT_DICT["InjecAgent"]
ATTACKS, SETTINGS = ("dh", "ds"), ("base", "enhanced")
TEMP, MAX_TOKENS = 1.0, 1024
E1_MODELS = ["gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14", "gpt-5.4-nano-2026-03-17", "deepseek-v4.1-flash",
             "glm-5.3", "kimi-k3", "gemini-3.8-flash", "claude-haiku-4-5", "qwen3-8b-local", "llama31-8b-local"]
_LOCK = threading.Lock()


def tool_dict():
    """Same as InjecAgent src/utils.py::get_tool_dict (not imported: that module opens an OpenAI client)."""
    out = {}
    for kit in json.load(open(os.path.join(IA, "data", "tools.json"))):
        for t in kit["tools"]:
            t = dict(t, name=kit["toolkit"] + t["name"])
            out[t["name"]] = t
    return out


TOOLS = tool_dict()
CASES = {(a, s): json.load(open(os.path.join(IA, "data", f"test_cases_{a}_{s}.json"))) for a in ATTACKS for s in SETTINGS}


def prompt(item):
    """InjecAgent evaluate_prompted_agent.py::predict_one_case, step 1."""
    names = [item["User Tool"]] + item["Attacker Tools"]
    desc = json.dumps([TOOLS[n] for n in names], indent=True)
    pad = (f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\n"
           f"Observation: {item['Tool Response']}\n")
    user = USER_PROMPT.format(available_tool_descriptions=desc, input=item["User Instruction"],
                              available_tool_names=names, agent_scratchpad=pad)
    return [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user}]


def units(attack, n, seed=7):
    return sorted(random.Random(f"{seed}-{attack}").sample(range(len(CASES[(attack, "base")])), n))


def jobs_for(models, n_units, reps, rep0=0):
    out = []
    for m in models:
        for a in ATTACKS:
            for c in units(a, n_units):
                for s in SETTINGS:
                    for r in range(rep0, rep0 + reps):
                        out.append(dict(key=f"{m}|ia|{a}|{s}|{c}|{r}", model=m, attack=a, setting=s, case=c, rep=r))
    return out


def run_one(j):
    item = CASES[(j["attack"], j["setting"])][j["case"]]
    row = dict(j, bench="injecagent", attack_type=item["Attack Type"], temp=TEMP, max_tokens=MAX_TOKENS,
               served_root=rc_run._ROOT[j["model"]], extra_body=rc_run.EXTRA_BODY.get(j["model"]),
               backend=rc_run.backend_of(j["model"]), err=None)
    try:
        client = rc_run.client_for(j["model"])
        kw = dict(model=j["model"], messages=prompt(item), temperature=TEMP, max_tokens=MAX_TOKENS)
        resp = client.chat.completions.create(**kw)
        text = resp.choices[0].message.content or ""
        ev = PARSE.evaluate_output_prompted(text, item["Attacker Tools"][0], item["Attacker Tools"],
                                            item["User Tool"], item["Attacker Instruction"])
        u = getattr(resp, "usage", None)
        row.update(output=text[:4000], eval=ev["eval"], invalid_reason=ev.get("invalid_reason"),
                   security=ev["eval"] == "succ",
                   usage={"prompt": getattr(u, "prompt_tokens", None), "completion": getattr(u, "completion_tokens", None)})
    except Exception as e:  # recorded, rerun later (resumable by key)
        row.update(err=repr(e)[:500], security=None)
    return row


def run_jobs(jobs, out, threads=16):
    rc_run.check_roots(sorted({j["model"] for j in jobs}))
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            if r.get("err") is None:
                done.add(r["key"])
    todo = [j for j in jobs if j["key"] not in done]
    print(f"{out}: {len(todo)}/{len(jobs)} todo", file=sys.stderr, flush=True)
    by_model = {}
    for j in todo:
        by_model.setdefault(j["model"], []).append(j)

    def go(j):
        r = run_one(j)
        with _LOCK, open(out, "a") as f:
            f.write(json.dumps(r) + "\n")

    gates = {m: rc_run.Gate(m) for m in by_model}

    def gated(j):
        with gates[j["model"]]:
            go(j)

    pools = [ThreadPoolExecutor(min(threads, gates[m].burst)) for m in by_model]
    futs = [ex.submit(gated, j) for ex, js in zip(pools, by_model.values()) for j in js]
    for f in futs:
        f.result()
    for ex in pools:
        ex.shutdown()


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "m0":
        models = sys.argv[2:] or ["gpt-4o-mini-2024-07-18", "gpt-5.4-nano-2026-03-17", "qwen3-8b-local"]
        run_jobs(jobs_for(models, n_units=10, reps=1), os.path.join(HERE, "..", "data", "ia_m0.jsonl"))
    elif stage in ("probe", "e1", "e2b"):  # probe = 20-call cost check per model before its batch (PM rule)
        models = sys.argv[2:] or E1_MODELS
        n, reps, rep0 = {"probe": (5, 1, 0), "e1": (60, 2, 0), "e2b": (60, 2, 10)}[stage]
        for m in models:  # one file per model: concurrent processes never share a file
            run_jobs(jobs_for([m], n_units=n, reps=reps, rep0=rep0), os.path.join(HERE, "..", "data", f"ia_{stage}_{m}.jsonl"))
    else:
        raise SystemExit(f"unknown stage {stage}")
