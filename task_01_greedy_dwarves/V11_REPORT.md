# Task 1 — Жадные гномы: V11

## Purpose

V11 is a behavior-preserving optimization of the user's original five-policy, set-based solution. It is not a new strategy and has not been submitted to the official judge.

## Preserved decision logic

- Five opponent strategies, in the original order:
  1. value;
  2. value / mass;
  3. negative mass;
  4. value / sqrt(mass);
  5. value - 2 * mass.
- Three own continuation strategies: policy indices 1, 2, and 3.
- Candidate objective: 0.7 * maximum continuation result + 0.3 * mean continuation result.
- Original tie-break terms: +0.001 * value - 0.0001 * mass.
- Multiplicative model updates: 1.25 for a match, 0.85 otherwise, with the 1e-6 floor.
- Passes (action 0) are included in the model update as actual = -1.
- Opponent actions are processed sequentially, removing each selected item before predicting the next opponent.

## Optimization

The original repeated scan through a Python set and sorted policy order is replaced inside simulations and opponent prediction by policy-specific rank bitmasks and cumulative capacity masks. The lowest set bit identifies the first feasible item in the same sorted policy order.

The available set is deliberately retained for top-level candidate enumeration so the original set iteration and tie-breaking order are preserved.

## Differential validation

Local tests compare V11 against a literal set-based reference implementation:

- 41,655 simulated candidate/policy/continuation outcomes compared: exact match.
- 12 randomized full games: exact same own decisions at every turn.
- 142 own decisions compared.
- 52 explicit opponent-pass observations included.
- After every round, available items, player masses, player values, and normalized strategy weights matched.
- Interactive subprocess protocol: 16 turns, five sequential bot actions per turn, legal own actions, clean exit: PASS.

## Performance sample

Environment: CPython 3.13.5. Python 3.11 / PyPy was not available in the local test container.

Maximum parameter case: K=10, N=100, T=20, W=50; one first-move decision on the same generated instance:

| Implementation | Measured time |
|---|---:|
| Original set-based reference | 1.35–1.48 s |
| V11 bitmask implementation | 0.072–0.076 s |
| Observed speedup | about 18–20x |

This is one local synthetic benchmark, not an official judge runtime or score guarantee.

## Official score

Not submitted yet. Preserve the original user-supplied code as a rollback baseline. The next useful experiment is to submit V11 unchanged and record its official score before changing any strategy parameters.
