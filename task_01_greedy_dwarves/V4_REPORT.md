# Task 1 — Greedy Dwarves V4 report

## Official-score status

V4 has not yet been submitted to the official judge. The official V3 score reported by the user was 4.189, below the previous 4.3 result. Local experiments do not establish an official-score improvement.

## Algorithmic change

V3 expanded the policy set from 10 to 46 and increased the number of own-continuation policies to 46. That made each candidate evaluation require up to 2,116 rollouts, versus 100 in V2. At the maximum case, the local V2 first move took about 0.10 CPU seconds, while the per-turn allocation with a 7.5-second budget over 20 rounds is 0.3375 seconds. A 21x increase can therefore cause the solver to stop before evaluating all legal candidates.

V4 returns to a compact 10-policy opponent model and changes the continuation objective:

- fixed_value: best expected score when one continuation policy must be chosen before the hidden bot strategy is known;
- adaptive_value: expected best continuation after the next batch of opponent actions provides more evidence;
- final candidate score: 0.25 * fixed_value + 0.75 * adaptive_value.

The opponent posterior uses a sharper rank likelihood (exp(-1.5 * rank)) than V2. The shared-strategy assumption is preserved.

## Local benchmark

All games use one common deterministic hidden strategy for all opponents within each test. The same generated item sets and hidden strategies were used for both versions.

### In-model benchmark

10 item sets, K=5, N=60, T=15, W=35, 10 hidden strategies, 100 games per version:

| Version | Mean raw reward | Positive games |
|---|---:|---:|
| V2 baseline (alpha=0, beta=0.8) | 270.05 | 100/100 |
| V4 adaptive (alpha=0.75, beta=1.5) | **284.92** | 100/100 |

Mean raw reward increased by approximately 5.5% in this synthetic benchmark.

### Expanded-policy out-of-model benchmark

10 item sets, 7 hidden policies sampled across the expanded 46-policy library, 70 games per version:

| Version | Mean raw reward | Positive games |
|---|---:|---:|
| V2 baseline | 259.29 | 64/70 |
| V4 adaptive | **279.01** | 66/70 |

Mean raw reward increased by approximately 7.6% in this sample.

These are synthetic benchmarks, not official contest scores. The hidden strategy used by the official judge is unknown.

## Correctness and performance checks

- Python 3.11 compilation: PASS.
- 100 randomized local games with legal shared-policy opponents: PASS.
- Forced-pass edge case: PASS.
- Posterior normalization check: PASS.
- Subprocess interactive protocol: 5/5 games, exactly 20 actions each, no extra stdout, exit code 0.
- Maximum-size first decision (K=10, N=100, T=20, W=50): about 0.10 CPU seconds on the local test environment; per-turn budget at startup is 0.3375 seconds.

## Next step

Submit solution_v4.py to the official judge and compare its score with V3 (4.189) and the previous best (4.3). Keep V2/V3 unchanged as rollback baselines.
