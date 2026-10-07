# E-C: replay comparison (K = 27, alpha = 0.05, 200 replays x 1000 rounds)

## real

| method | FWER | correct @100 | correct @250 | correct @500 | correct @1000 | mean round |
|---|---|---|---|---|---|---|
| comparison (ours) | 0.000 | 3.25 | 15.93 | 19.95 | 22.91 | 250.0 |
| sign (ours) | 0.000 | 12.90 | 18.55 | 21.66 | 22.82 | 155.2 |
| Rank CS | 0.000 | 9.85 | 17.70 | 21.70 | 22.98 | 177.6 |
| BB-EDGE | 0.000 | 4.58 | 15.71 | 19.27 | 22.68 | 251.8 |

## null

| method | FWER | correct @100 | correct @250 | correct @500 | correct @1000 | mean round |
|---|---|---|---|---|---|---|
| comparison (ours) | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | None |
| sign (ours) | 0.015 | 0.00 | 0.00 | 0.00 | 0.00 | None |
| Rank CS | 0.005 | 0.00 | 0.00 | 0.00 | 0.00 | None |
| BB-EDGE | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | None |

C-valid: {'comparison (ours)': True, 'sign (ours)': True, 'Rank CS': True, 'BB-EDGE': True}
C-power: {'sign_at_500': 21.66, 'rank_at_500': 21.695, 'bb_at_500': 19.265, 'pass': False}
