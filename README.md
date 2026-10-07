# RiskCert: anytime-valid risk certificates for LLM agent security audits

Code, frozen experiment data and analysis scripts for the RiskCert paper. Every number, table and figure in the paper is
regenerated from the released per-episode records by the scripts below.

## Layout
| path | content |
|---|---|
| `code/riskcert.py` | the certificates: betting confidence sequences (plug-in, RC-mix), paired comparison, discordant-pair sign process, decision rules |
| `code/*_run.py`, `testbed/` | experiment runners (AgentDojo through `testbed/harness.py`, InjecAgent through `code/ia_run.py`) |
| `code/*_analyze.py`, `code/*_replay.py`, `code/fixed_*.py`, `code/eb_baselines.py`, `code/ec_compare.py`, `code/x_positioning.py` | preregistered analyses, replays and simulations |
| `code/make_numbers.py`, `code/make_figs.py` | write `paper_artifacts/numbers.tex`, `paper_artifacts/tab_*.tex` and `paper_artifacts/figs/*.pdf` |
| `code/test_*.py` | synthetic tests (no model calls) |
| `data/` | frozen per-episode records (`*.jsonl`), the frozen unit list and payload pool with SHA-256 files |
| `results/` | analysis outputs (`*.json`, `*.md`), result-audit inputs (`audit_*/`) |
| `paper_artifacts/` | the generated macros, tables and figures used by the paper |
| `PREREGISTRATION.md` | the experiment plans, frozen before each experiment ran |
| `PRIVACY.md` | what was removed or rewritten for the public release |

## Setup
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```
Python 3.11 or newer. The analysis part needs only numpy, scipy, matplotlib and pytest.

## Reproduce the paper's numbers, tables and figures (no model calls)
```bash
cd code
python3 make_numbers.py      # -> paper_artifacts/numbers.tex and paper_artifacts/tab_*.tex
python3 make_figs.py         # -> paper_artifacts/figs/*.pdf
python3 -m pytest -q         # synthetic tests; InjecAgent tests are skipped without a checkout
```
The released `paper_artifacts/` were produced this way; `make_numbers.py` regenerates them byte for byte from `results/`.

To recompute the analyses themselves from the episode records:
| paper element | command | output |
|---|---|---|
| RQ1 channel audit and replication | `python3 c3_analyze.py`, `C3_PREFIX=c3r C3_SEED_BASE=51 python3 c3_analyze.py` | `results/c3.json`, `results/c3r.json` |
| RQ1 fixed-design confirmation, session-B certificates | `python3 fixed_design.py` | `results/m3c.json` |
| RQ2 coverage replays | `python3 sessb_analyze.py rq2` | `results/rq2.json` |
| RQ2 fixed-grid simulation | `python3 fixed_grid_sim.py 1000` | `results/fixed_grid_sim.json` |
| RQ3 interim looks and selection | `python3 sessb_analyze.py e3p` | `results/e3p.json` |
| RQ3 baselines with the same contract | `python3 eb_baselines.py 1000` | `results/eb_baselines.json` |
| RQ3 replay comparison with Rank CS and BB-EDGE | `python3 ec_compare.py 200 1000` | `results/ec_compare.json` |
| RQ4 release audit replays | `python3 e4_replay.py`, then `python3 m3b_analyze.py e4` | `results/e4_replay_*.json`, `results/e4.json` |
| RQ4 discovery stop | `python3 m3b_analyze.py e5` | `results/e5.json` |
| positioning experiments X1–X3 | `python3 x_positioning.py x1 500`, `x2 1000`, `x3` | `results/x1.json`, `results/x2.json`, `results/x3.json` |

Macro groups in `numbers.tex` and their sources: `\cOne*`, `\cOneB*` (results/c1.json, c1b.json, m3c.json), `\cTwo*`, `\rqOne*`, `\rqTwo*`, `\eThree*` (c2.json, rq2.json, e3p.json), `\eFour*`, `\eFive*` (e4_replay_*.json, e5.json), `\sim*` (fixed_grid_sim.json), `\cThree*`, `\cThreeR*` (c3.json, c3r.json), `\eb*`, `\ec*` (eb_baselines.json, ec_compare.json), `*Upper` (Clopper–Pearson bounds from the same files).

## Run new episodes (optional)
Live runs call models through any OpenAI-compatible endpoint and self-hosted models through vLLM on localhost ports.
```bash
export API_BASE_URL=https://<your-endpoint>/v1 API_KEY=<your-key>
git clone https://github.com/uiuc-kang-lab/InjecAgent third_party/InjecAgent   # InjecAgent experiments only
cd code && python3 c3_run.py probe gpt-5.4-nano-2026-03-17
```
Local models are mapped to ports in `code/rc_run.py` (`LOCAL`). Model outputs are stochastic (temperature 1), so new runs reproduce the protocol, not the exact records.

## Third-party code
AgentDojo (MIT) is a pip dependency. InjecAgent (MIT) is cloned separately into `third_party/InjecAgent`. Neither is redistributed here.

## License
MIT, see `LICENSE`.
