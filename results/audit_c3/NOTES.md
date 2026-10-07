# C3 result-audit notes (2026-10-07)

- audit.py: 0 FAIL, 13 OK (recompute 8/8 exact, polarity 73/73 no-tool-call rows incl. 60 null-agent anchors, provenance 4 served ids).
- With `--plan plan.md` the heuristic plan check FAILs on "n=150": that number is from the power paragraph, while C3 uses a preregistered sequential stop (variable n per model). Samples = 2 x (36+102+45+450) used rows + 60 null rows = 1326, as the plan specifies. Not a load-bearing failure.
- audit_prep c3 asserts every used row's unit and position equal the frozen unit list (sha 65eff35c...) at the seeded draw (seed 31+idx).
- GLM and DeepSeek were restarted once with C3_BATCH=12 (resumable, rounds processed in draw order); c3_analyze reproduces the runner's stop rounds (36/102/45/450).
- Post-hoc (not preregistered, PM ruling: not in the main text): R1 (ContrastCS + RC-mix, a = 0.05/4) read at each model's stop certifies none of the four contrasts (see `r1_posthoc_*` in results/c3.json). The EB robustness interval also certifies none at the stops. Only the preregistered sign certificate (R1') decides, from the discordant rounds.
- af1 xcheck (paper-tripwire/review/xcheck_riskcert_c3.md): reproduced, no SEVERE. c3_analyze.py was edited once during collection (02:39 run-local time) to add the per-round `trace` field only; decisions unchanged. c3_run docstring corrected: over-run rows carry no `used` field, they are excluded because c3_analyze reads only rounds before the stop.
