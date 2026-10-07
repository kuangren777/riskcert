# C3R result-audit notes (preregistered 5d6cc121; erratum withdrawn e7abf5b6)

- audit.py: 0 FAIL, 11 OK (recompute exact, polarity 75/75 no-tool-call rows incl. 60 C3 null-agent anchors, provenance 3 served ids).
- R-each: all 3 models certify the C3 direction (Qwen3-8B tr>cfg at round 62, GLM-5.3 cfg>tr at 36, GPT-5.4-nano cfg>tr at 47). R-P2 PASS.
- Error handling identical to C3: Qwen round 30 tool_return hit a malformed FunctionCall (arguments as a string) 3 times in two runs (2 err rows), then a valid episode on the third resume. The withdrawn E5-style rule was never applied (0 rows with err_kind). No other error rows.
- c3_analyze reproduces the runner's stop rounds (62/36/47).
