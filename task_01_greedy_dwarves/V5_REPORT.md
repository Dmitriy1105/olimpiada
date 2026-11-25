# Task 1 — Greedy Dwarves V5

## Baseline

The user-supplied solution_best_4_3.py is preserved unchanged. Its reported official score is 4.3. V3 (4.189) and V4 (3.84) are not used as the new baseline.

## V5 changes

V5 preserves the best solution's central decision objective:
- three own continuation policies: value/mass, minimum mass, value/sqrt(mass);
- 70% maximum continuation result + 30% mean continuation result;
- one shared posterior over the hidden strategy used by all opponents;
- same item tie-break terms.

Only the opponent model and implementation are changed:
- retain the original five policy hypotheses;
- add three targeted hypotheses: value - 1.5*mass, value - 3*mass, value/(mass+5);
- replace the binary 1.25/0.85 weight update with a soft-rank Bayesian likelihood;
- use rank bitmasks and capacity masks to speed up repeated simulations.

The soft-rank likelihood is:
P(observed | policy) = (1-epsilon) * exp(-beta*rank)/Z + epsilon/L,
where rank is the number of currently feasible items preferred to the observed item, L is the number of currently feasible items, beta=0.7, epsilon=0.02.

## Local synthetic benchmark

The baseline is the supplied five-policy algorithm with its original 1.25/0.85 updates. Both algorithms are evaluated on identical item sets and deterministic shared opponent policies. K=4, N=45, T=12, W=30.

| Hidden opponent policy family | Games | Baseline mean reward | V5 mean reward | Relative change |
|---|---:|---:|---:|---:|
| Original five policies | 100 | 225.68 | 232.35 | +2.96% |
| Three added policies | 60 | 203.97 | 214.55 | +5.19% |

All 160 games were positive for both algorithms in this test sample. These are synthetic tests, not official contest scores; the hidden jury strategy is unknown.

## Correctness and performance

- Bitmask simulator compared with a set-based reference on 300 randomized states: PASS.
- 100 randomized complete games with one shared hidden opponent policy: PASS.
- Posterior normalization: PASS.
- Sample interactive protocol: 5 rounds completed with legal moves.
- Maximum constraints K=10, N=100, T=20, W=50: local in-process CPU time approximately 0.11 seconds for a full simulated game in the test environment.

## Official score

Not yet submitted. Compare the official V5 result against the preserved 4.3-point baseline.
