# Task 1 — Жадные гномы: V12

## Change from V11

V12 is a single-variable experiment. It preserves the V11 five-opponent-policy model, bitmask simulator, set-based candidate iteration, 1.25/0.85 multiplicative updates (including pass actions), 70/30 objective, and tie-break terms.

The only algorithmic change is the set of own continuation policies:

- V11: (1, 2, 3) — value/mass, minimum mass, value/sqrt(mass).
- V12: (0, 1, 2, 3) — adds the maximum-value policy as a fourth continuation hypothesis.

The score remains 0.7 * max(results) + 0.3 * mean(results), with the mean now taken over four own continuation policies.

## Local synthetic benchmark

100 deterministic generated games. All opponents in each game used the same hidden one of the five greedy policies, matching the shared-hidden-policy assumption. V11 and V12 were run on identical seeds and item sets.

| Metric | V11 | V12 |
|---|---:|---:|
| Mean reward | 248.00 | 249.52 |
| Positive-reward games | 100/100 | 100/100 |
| Games where V12 reward was higher | — | 16 |
| Games where V12 reward was lower | — | 12 |
| Equal-reward games | — | 72 |

The mean synthetic reward increased by about 0.61%. This is a small local sample and does not predict or guarantee an official score improvement.

## Validation

- Python syntax compilation: PASS (local interpreter).
- 100 complete deterministic shared-policy games: PASS.
- V12 changes only the own continuation-policy tuple; the opponent model and action protocol remain unchanged.
- Official judge result: not yet submitted / unknown.

## Submission guidance

Keep V11 unchanged as the 4.43-point rollback version. Submit V12 as a separate experiment and compare its official score. Do not overwrite V11.
