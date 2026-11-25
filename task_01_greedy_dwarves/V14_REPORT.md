# Task 1 — Жадные гномы: V14

## Official-score history

- V12: **4.676** (user-reported official score; current best).
- V13: **4.326** (user-reported official score; lower than V12).
- V14: not yet submitted; official score unknown.

V12 remains the rollback baseline. V13 is retained as a failed official experiment.

## Single change from V12

V14 changes only the multiplicative opponent-model weight updates:

| Observation | V12 | V14 |
|---|---:|---:|
| Predicted action matches observed action | ×1.25 | **×2.0** |
| Prediction does not match | ×0.85 | **×0.5** |

The five opponent-policy hypotheses, four own continuation policies, 70/30 max/mean objective, pass handling, action order, bit-mask simulator, candidate iteration order, and tie-break terms are unchanged.

The stronger update makes the shared opponent-strategy model concentrate more quickly on policies that predict observed actions.

## Deterministic paired synthetic benchmarks

All opponents in each game use one shared hidden deterministic policy. V12 and V14 play the same instances against the same hidden policy. The normalized metric is reward divided by the optimal 0/1-knapsack value subject to capacity and at most T selected items.

### Randomized task parameters — 300 games

K=3..10, N=10..100, T=5..20, W=10..50. The hidden strategy is balanced across the five V12 policy hypotheses.

| Metric | V12 | V14 |
|---|---:|---:|
| Mean raw reward | 211.317 | **213.570** |
| Mean normalized reward | 0.541547 | **0.546925** |
| Positive-reward games | 298/300 | 298/300 |

Paired comparison: V14 wins 52, loses 14, ties 234. Mean normalized delta: **+0.0053778**.

### Fixed parameters — 300 games

K=5, N=60, T=15, W=35. Thirty games per each of the five shared hidden policies.

| Metric | V12 | V14 |
|---|---:|---:|
| Mean raw reward | 273.407 | **276.063** |
| Mean normalized reward | 0.580207 | **0.585579** |
| Positive-reward games | 300/300 | 300/300 |

Paired comparison: V14 wins 89, loses 38, ties 173. Mean normalized delta: **+0.0053717**.

### Expanded deterministic hidden-policy pool — 200 games

The hidden strategy is selected from ten deterministic item rankings: the five V12 policies plus value/mass^0.7, value−1.5×mass, value−3×mass, value/(mass+5), and value/(mass+10).

| Metric | V12 | V14 |
|---|---:|---:|
| Mean raw reward | 210.565 | **213.375** |
| Mean normalized reward | 0.539487 | **0.546155** |
| Positive-reward games | 199/200 | 200/200 |

Paired comparison: V14 wins 32, loses 20, ties 148. Mean normalized delta: **+0.0066679**.

## Stochastic-opponent sensitivity check

A separate 100-game fixed-parameter stress test allowed each opponent, with a shared stochastic rule, to choose randomly among its top three feasible items with a specified probability.

| Noise probability | V12 normalized | V14 normalized |
|---|---:|---:|
| 10% | 0.550648 | **0.563663** |
| 25% | **0.544941** | 0.538943 |

This indicates that stronger updates can be less suitable if the hidden strategy is substantially stochastic. The task statement says opponents in a test share one strategy, but does not establish whether that strategy is deterministic. Therefore these synthetic results do not establish an expected official-score improvement.

## Protocol validation

- Python syntax compilation: PASS.
- Maximum-size subprocess test: K=10, N=100, T=20, W=50.
- All 20 actions were received and legal.
- Process exit code: 0.
- Extra stdout: none.
- stderr: empty.
- Local runtime: 2.40 seconds on CPython 3.13.5.

The task runtime limit is 10 seconds. Python 3.11 was not available locally; the implementation uses standard-library syntax compatible with Python 3.11.

## Recommendation

Keep V12 unchanged as the best official version (4.676). Submit V14 as a separate experiment. Do not replace the best version until the official judge score is known.
