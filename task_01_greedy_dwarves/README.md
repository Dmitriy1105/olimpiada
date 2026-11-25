# Task 1 — Жадные гномы

Интерактивная задача, Python 3.11+.

## Versions

- `solution_best_4_3.py` — exact user-supplied baseline; reported official score 4.3.
- `solution_v5.py` — experimental V5 candidate.
- `solution_v3.py` — earlier 46-policy experiment.
- `solution_v4.py` — earlier adaptive-continuation experiment.
- `test_solution_v5.py` — simulator equivalence and randomized protocol tests.
- `V5_REPORT.md` — benchmark and validation report.

## V5

V5 preserves the best baseline's three continuation policies and the 70% maximum + 30% mean objective. It adds three targeted opponent-policy hypotheses, replaces the binary posterior update with a soft-rank Bayesian likelihood, and uses bitmasks to accelerate simulation.

Local synthetic results are not official contest scores. The official V5 result is not yet known.
