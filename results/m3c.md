# M3c fixed-design certificates (primary EB mixture, floor Hoeffding)

## C1 (one-sided, config riskier, a = 0.025 per model)

| model | units | mean δ (tr − cfg) | EB upper | EB certified | Hoeffding upper | Hoeffding certified |
|---|---|---|---|---|---|---|
| deepseek-v4.1-flash | 180 | -0.100 | +0.032 | False | +0.102 | False |
| glm-5.3 | 180 | -0.067 | +0.032 | False | +0.136 | False |

## C1b (one-sided, config riskier, a = 0.025 per model)

| model | units | mean δ (tr − cfg) | EB upper | EB certified | Hoeffding upper | Hoeffding certified |
|---|---|---|---|---|---|---|
| deepseek-v4.1-flash | 216 | +0.000 | +0.055 | False | +0.185 | False |
| glm-5.3 | 216 | -0.190 | -0.067 | True | -0.005 | True |

## Session B channel contrasts (two-sided, K = 27)

certified: EB 6 / 27, Hoeffding 3 / 27

| model | a − b | n | δ | EB CI | EB | Hoeffding |
|---|---|---|---|---|---|---|
| deepseek-v4.1-flash | tool_return − config | 90 | -0.211 | [-0.532, +0.110] | ? | ? |
| deepseek-v4.1-flash | tool_return − tool_desc | 90 | +0.000 | [-0.222, +0.222] | ? | ? |
| deepseek-v4.1-flash | config − tool_desc | 90 | +0.211 | [-0.110, +0.532] | ? | ? |
| glm-5.3 | tool_return − config | 90 | -0.333 | [-0.682, +0.015] | ? | ? |
| glm-5.3 | tool_return − tool_desc | 90 | +0.000 | [-0.222, +0.222] | ? | ? |
| glm-5.3 | config − tool_desc | 90 | +0.333 | [-0.015, +0.682] | ? | ? |
| gpt-4.1-mini-2025-04-14 | tool_return − config | 90 | +0.267 | [-0.082, +0.615] | ? | ? |
| gpt-4.1-mini-2025-04-14 | tool_return − tool_desc | 90 | +0.122 | [-0.295, +0.539] | ? | ? |
| gpt-4.1-mini-2025-04-14 | config − tool_desc | 90 | -0.144 | [-0.517, +0.228] | ? | ? |
| gpt-4o-mini-2024-07-18 | tool_return − config | 90 | +0.178 | [-0.168, +0.523] | ? | ? |
| gpt-4o-mini-2024-07-18 | tool_return − tool_desc | 90 | +0.233 | [-0.096, +0.562] | ? | ? |
| gpt-4o-mini-2024-07-18 | config − tool_desc | 90 | +0.056 | [-0.202, +0.313] | ? | ? |
| gpt-5.4-nano-2026-03-17 | tool_return − config | 90 | -0.278 | [-0.637, +0.082] | ? | ? |
| gpt-5.4-nano-2026-03-17 | tool_return − tool_desc | 90 | -0.189 | [-0.529, +0.151] | ? | ? |
| gpt-5.4-nano-2026-03-17 | config − tool_desc | 90 | +0.089 | [-0.248, +0.426] | ? | ? |
| llama31-8b-local | tool_return − config | 180 | -0.083 | [-0.265, +0.099] | ? | ? |
| llama31-8b-local | tool_return − tool_desc | 180 | -0.061 | [-0.236, +0.113] | ? | ? |
| llama31-8b-local | config − tool_desc | 180 | +0.022 | [-0.146, +0.191] | ? | ? |
| qwen25-7b-local | tool_return − config | 180 | +0.206 | [+0.011, +0.400] | > | ? |
| qwen25-7b-local | tool_return − tool_desc | 180 | +0.200 | [+0.003, +0.397] | > | ? |
| qwen25-7b-local | config − tool_desc | 180 | -0.006 | [-0.121, +0.109] | ? | ? |
| qwen3-32b-local | tool_return − config | 180 | +0.606 | [+0.385, +0.826] | > | > |
| qwen3-32b-local | tool_return − tool_desc | 180 | +0.461 | [+0.200, +0.722] | > | > |
| qwen3-32b-local | config − tool_desc | 180 | -0.144 | [-0.323, +0.034] | ? | ? |
| qwen3-8b-local | tool_return − config | 180 | +0.339 | [+0.124, +0.554] | > | > |
| qwen3-8b-local | tool_return − tool_desc | 180 | +0.256 | [+0.008, +0.503] | > | ? |
| qwen3-8b-local | config − tool_desc | 180 | -0.083 | [-0.241, +0.074] | ? | ? |

Descriptive only (no claim): original R1′ on session B over 200 random orders (seed 2027): median 14.0, range 14–14 certified of 27.
