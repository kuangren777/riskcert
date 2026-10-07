# E3′: FWER under peeking and post hoc selection (preregistered, session-B pools)

K = 27, replays = 400, rounds = 500; pass = R1 and R1′ FWER ≤ 0.083 in every setting and L

## null

| method | L=1 | L=5 | L=20 |
|---|---|---|---|
| R1 | 0.003 | 0.000 | 0.003 |
| R1prime | 0.003 | 0.013 | 0.013 |
| SPRT0.05 | 0.675 (in-zone 0.675) | 0.980 (in-zone 0.980) | 0.995 (in-zone 0.995) |
| SPRT0.01 | 0.487 (in-zone 0.487) | 0.882 (in-zone 0.882) | 0.958 (in-zone 0.958) |
| McNemar_peek | 0.040 | 0.068 | 0.133 |
| Wilson_peek | 0.050 | 0.100 | 0.212 |

## real

| method | L=1 | L=5 | L=20 |
|---|---|---|---|
| R1 | 0.000 | 0.000 | 0.000 |
| R1prime | 0.000 | 0.000 | 0.000 |
| SPRT0.05 | 0.000 (in-zone 0.000) | 0.000 (in-zone 0.000) | 0.000 (in-zone 0.000) |
| SPRT0.01 | 0.000 (in-zone 0.000) | 0.000 (in-zone 0.000) | 0.000 (in-zone 0.000) |
| McNemar_peek | 0.000 | 0.000 | 0.000 |
| Wilson_peek | 0.000 | 0.000 | 0.000 |

**Verdict (preregistered): PASS**
