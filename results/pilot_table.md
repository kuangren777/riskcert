# pilot cell table (auto, rc_analyze.py)

valid rows 2565, err rows 0

served roots: gpt-4.1-mini-2025-04-14=api-gateway:gpt-4.1-mini-2025-04-14; gpt-4o-mini-2024-07-18=api-gateway:gpt-4o-mini-2024-07-18; gpt-5.4-nano-2026-03-17=api-gateway:gpt-5.4-nano-2026-03-17; llama31-8b-local=meta-llama/Llama-3.1-8B-Instruct; qwen3-8b-local=Qwen/Qwen3-8B

think-stop rows (empty trace, final starts with <think>): gpt-4.1-mini-2025-04-14=0; gpt-4o-mini-2024-07-18=0; gpt-5.4-nano-2026-03-17=0; llama31-8b-local=0; qwen3-8b-local=0

| model | pooled | tool_r/sta | tool_r/mid | tool_r/end | config/sta | config/mid | config/end | tool_d/sta | tool_d/mid | tool_d/end |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4.1-mini-2025-04-14 | 239/540=0.44 | 40/60 | 31/60 | 38/60 | 19/60 | 18/60 | 24/60 | 25/60 | 25/60 | 19/60 |
| gpt-4o-mini-2024-07-18 | 44/540=0.08 | 17/60 | 9/60 | 8/60 | 4/60 | 0/60 | 1/60 | 1/60 | 1/60 | 3/60 |
| gpt-5.4-nano-2026-03-17 | 82/540=0.15 | 0/60 | 1/60 | 0/60 | 25/60 | 17/60 | 15/60 | 12/60 | 5/60 | 7/60 |
| llama31-8b-local | 45/540=0.08 | 0/60 | 3/60 | 0/60 | 6/60 | 6/60 | 6/60 | 6/60 | 10/60 | 8/60 |
| qwen3-8b-local | 68/405=0.17 | 25/45 | 15/45 | 12/45 | 0/45 | 0/45 | 0/45 | 4/45 | 6/45 | 6/45 |

| model | tool_return | config | tool_desc | utility |
|---|---|---|---|---|
| gpt-4.1-mini-2025-04-14 | 0.61 [0.53,0.67] (n=180) | 0.34 [0.27,0.41] (n=180) | 0.38 [0.32,0.46] (n=180) | 0.50 |
| gpt-4o-mini-2024-07-18 | 0.19 [0.14,0.25] (n=180) | 0.03 [0.01,0.06] (n=180) | 0.03 [0.01,0.06] (n=180) | 0.44 |
| gpt-5.4-nano-2026-03-17 | 0.01 [0.00,0.03] (n=180) | 0.32 [0.25,0.39] (n=180) | 0.13 [0.09,0.19] (n=180) | 0.40 |
| llama31-8b-local | 0.02 [0.01,0.05] (n=180) | 0.10 [0.06,0.15] (n=180) | 0.13 [0.09,0.19] (n=180) | 0.21 |
| qwen3-8b-local | 0.39 [0.31,0.47] (n=135) | 0.00 [0.00,0.03] (n=135) | 0.12 [0.07,0.18] (n=135) | 0.33 |

## replay cost
```
{
 "cells": 45,
 "tau": 0.2,
 "alpha": 0.05,
 "n_max": 60,
 "sprt_p0p1": [
  0.1,
  0.4
 ],
 "riskcert_mean_draws": 2545.208,
 "sprt_bonf_mean_draws": 1238.84,
 "fixed_n_per_cell": 69,
 "fixed_n_total": 3105,
 "replays": 500
}
```
