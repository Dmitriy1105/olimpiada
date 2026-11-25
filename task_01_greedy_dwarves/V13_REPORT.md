# Task 1 — Жадные гномы: V13

## Official result

V12 received 4.676 points, as reported by the user on 2026-11-25. V13 has not yet been submitted; its official score is unknown.

## Single change from V12

V13 retains the V12 code and changes only the coefficient that combines the best and average own continuation-policy simulations.

V12 uses 0.70 * max(results) + 0.30 * mean(results).

V13:
- progress = round_no / max(1, T - 1)
- alpha = 0.85 - 0.30 * progress
- expected = alpha * max(results) + (1 - alpha) * mean(results)

Thus alpha moves linearly from 0.85 in the first round to 0.55 in the final round. The five opponent policies, four own continuation policies, opponent weight updates, pass handling, candidate order, bit-mask simulator, and tie-break terms are unchanged.

## Paired local simulation

All opponents in each game use one shared hidden policy. V12 and V13 play identical item sets against the same hidden policy. The normalized comparison uses a 0/1 knapsack dynamic program to calculate the maximum value achievable under capacity and at-most-T item selections.

### Randomized parameter set — 120 games

Parameters are sampled within the task bounds: K=3..10, N=10..100, T=5..20, W=10..50. The hidden opponent policy is selected once per game from the five policies in the V12 model.

| Metric | V12 | V13 |
|---|---:|---:|
| Mean raw reward | 194.067 | 195.692 |
| Mean normalized reward | 0.549189 | 0.553166 |
| Positive-reward games | 119/120 | 119/120 |

Paired results: V13 wins 13, loses 0, ties 107. Mean normalized difference: +0.0039777.

### Balanced shared-policy set — 150 games

Fixed parameters K=5, N=60, T=15, W=35. Each of the five hidden policies is used in 30 games.

| Metric | V12 | V13 |
|---|---:|---:|
| Mean raw reward | 273.487 | 275.227 |
| Mean normalized reward | 0.591149 | 0.595128 |
| Positive-reward games | 150/150 | 150/150 |

Paired results: V13 wins 24, loses 8, ties 118. Mean normalized difference: +0.0039798.

A separate fullness-dependent alpha schedule was tested and did not outperform the simpler round-dependent schedule on this balanced set, so it was not included in V13.

## Protocol and syntax checks

- Python bytecode compilation: PASS (local CPython 3.13.5).
- Maximum-size subprocess protocol: PASS, K=10, N=100, T=20, W=50.
- All 20 actions were received; actions were legal; no voluntary pass occurred; process exit code 0; no extra stdout; stderr empty.

Python 3.11 was not available in the local container. The source uses only standard-library features supported by Python 3.11.

## Interpretation and next step

The adaptive blend improves both local paired datasets by about 0.004 in normalized reward, while changing no opponent-model assumptions. This is encouraging but remains a synthetic result, not a guarantee about hidden official tests.

V13 was submitted and received **4.326 points**, below V12's **4.676**. Keep V12 unchanged as the best official rollback. V13 remains archived as a failed official experiment.
