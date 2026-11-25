# Task 1 — Жадные гномы: V15

## Official score history

- V11: 4.43 (user-reported official score).
- V12: **4.676** (current best official score; user-reported).
- V13: 4.326 (user-reported official score).
- V14: approximately 4.3 (user-reported official score).
- V15: not yet submitted; official score unknown.

Keep V12 unchanged as the official rollback baseline.

## New algorithmic idea

V12 scores a candidate first action by simulating four fixed greedy continuation policies for our own dwarf. V15 uses V12's score only to create a shortlist of up to 14 plausible first actions. It then runs a two-round beam search over our future actions for each shortlisted first action and each of the five opponent-policy hypotheses.

The beam search:
1. Applies the candidate first action and simulates all opponents' moves in the current round.
2. At each of the next two rounds, branches over up to six diverse feasible actions proposed by the five ranking orders.
3. Simulates all opponents sequentially after each proposed action.
4. Uses value-maximizing and value-density rollouts to rank beam states.
5. Evaluates the surviving states with all four V12 own-continuation policies.
6. Aggregates the result over the current opponent-policy weights.

V15 retains V12's five opponent hypotheses, multiplicative weight updates (1.25/0.85), item ordering, pass handling, and terminal objective. This isolates the effect of searching our own future action sequence rather than changing the opponent model at the same time.

Search parameters: ROOT_LIMIT=14, BEAM_WIDTH=4, BRANCH_FACTOR=6, SEARCH_DEPTH=2.

## Paired local synthetic experiments

Each V12/V15 pair uses exactly the same item instance and the same hidden deterministic opponent policy. All opponents in a game share that policy and act sequentially. The normalized metric is the player's reward divided by the 0/1-knapsack optimum under the same capacity and at-most-T item limit, matching the task's per-test normalization.

### Randomized task parameters — 120 games

Seeds 81000–81119. K, N, T, W are sampled within task constraints. The five hidden ranking policies are balanced (24 games per policy).

| Metric | V12 | V15 |
|---|---:|---:|
| Mean raw reward | 205.192 | **212.942** |
| Mean normalized reward | 0.556502 | **0.572515** |
| Positive-reward games | 119/120 | 119/120 |

Paired comparison: V15 wins 47, loses 10, ties 63. Mean normalized difference: **+0.016013**.

### Fixed parameters — 100 games

Seeds 82000–82099. K=5, N=60, T=15, W=35. The five hidden policies are balanced (20 games each).

| Metric | V12 | V15 |
|---|---:|---:|
| Mean raw reward | 275.010 | **284.820** |
| Mean normalized reward | 0.588492 | **0.607992** |
| Positive-reward games | 100/100 | 100/100 |

Paired comparison: V15 wins 58, loses 11, ties 31. Mean normalized difference: **+0.019500**.

### Expanded hidden-policy pool — 100 games

Seeds 83000–83099. Fixed K=5, N=60, T=15, W=35. Ten deterministic ranking policies are balanced (10 games each): the five V12 hypotheses plus five alternative rankings based on value/mass^0.7, value−1.5×mass, value−3×mass, value/(mass+5), and value/(mass+10).

| Metric | V12 | V15 |
|---|---:|---:|
| Mean normalized reward | 0.545205 | **0.562710** |
| Positive-reward games | 98/100 | 99/100 |

Paired comparison: V15 wins 49, loses 20, ties 31. Mean normalized difference: **+0.017505**.

These are local synthetic results, not official judge scores. The simulated opponents use deterministic ranking policies; the real hidden judge strategy may differ.

## Runtime and protocol

- Local runtime: CPython 3.13.5.
- Maximum-parameter direct simulation: K=10, N=100, T=20, W=50; one full game completed in approximately 0.66 seconds.
- Subprocess interactive test at maximum parameters: 20 actions received, all actions legal, process exit code 0, no extra stdout.
- Python 3.11 / PyPy 7.3.19 was not available locally. The code uses standard-library features supported by Python 3.11.

## Recommendation

Submit V15 as a separate experiment. Do not overwrite V12. If V15 improves the official score, preserve it as the new baseline; if not, the next major direction should be a richer online model of the shared hidden opponent strategy rather than more coefficient tuning.
