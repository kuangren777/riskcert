# Session-B audit notes (af5, 2026-10-06)

- DeepSeek-V4.1-Flash session-B config successes are 19/90, the same count as session A (E1 controls).
  Checked for contamination: session-B keys use reps 10-11 against session A's 0-1, the key overlap is 0, GLM changed from 19 to 30, and only 7 of the 19 successful (pos, suite, ut, it) units recur.
  Conclusion: coincidence, not contamination. This stays out of the paper text.
- 16k context-overflow rows: 3 Qwen2.5-7B keys were completed by reruns (2 on the first rerun, 1 on the second; reruns then stopped).
  Overflow correlates with episode length, so completing by rerun can favour shorter episodes. Sensitivity (`results/sessb_sensitivity.md`): with the keys dropped or counted as safe, C2, RQ2 cross-session and E3' real are all unchanged.
- Error-buffer use in total: 36 of 300 (C1b 17, session-B hub 15, Qwen2.5 4).
- Polarity anchor: 330 rows without tool calls (135 null-agent + 195 natural), 0 violations.
- Independent RQ2 recompute by af1 with another seed (numpy default_rng 123, 200 replays, InjecAgent ASR-all, K=48):
  the plug-in interval missed once (replay 9, cell gpt-4o-mini|ds|enhanced, step 26: 18/26 successes, lower bound 0.296 > pool truth 0.2833).
  1/200 = 0.005 ≤ α, so this is consistent with Theorem 1. The validity of the plug-in bet is proved: the bet is predictable, the hedge is fixed and grid closure is applied. Conjecture 2.4 concerns only its sample-size rate.
  The paper reports the preregistered run's measured proportion (0 of 200), not "never misses".
