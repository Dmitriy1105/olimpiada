# Benchmark report — Task 1 V3

## Изменения

V3 replaces 10 fixed hypotheses with 46 hypotheses:
- 1 pure-value strategy;
- 11 power-density strategies;
- 16 linear mass-penalty strategies;
- 9 shifted-density strategies;
- 7 quadratic mass-penalty strategies;
- 2 mass-extreme strategies.

One shared posterior is maintained for all opponents in a test.

## Local comparative experiment

Synthetic benchmark: 10 seed sets; K=5, N=60, T=15, W=35; 20 deterministic hidden policies per seed; all opponents in each test use one shared policy; both versions see identical items and hidden policies.

| Version | Mean raw reward | Positive games |
|---|---:|---:|
| V2 baseline (10 hypotheses) | 259.64 | 191/200 |
| V3 (46 hypotheses) | 279.42 | 200/200 |

V3 mean reward is approximately 7.6% higher in this particular synthetic experiment.

## Additional prior sensitivity check

On 18 games where the hidden policy was selected uniformly from six strategy families, a family-balanced prior scored 273.33 mean reward versus 269.17 for the uniform prior. On a separate 30-game sample using the original ten policy types, the family-balanced prior scored 288.23 versus 306.47 for the uniform prior. The main V3 therefore retains the uniform prior, which performed better on the broader 200-game comparison.

## Protocol and limitations

- A local interactive protocol smoke test completed without a protocol violation.
- The protocol test used K=10, N=100, T=20, W=50.
- Local simulations use synthetic policies; they are not official jury results.
- The jury strategies are unknown, so an official score increase is not guaranteed.