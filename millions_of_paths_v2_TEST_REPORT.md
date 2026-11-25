# Millions of Paths — version 2 validation

Date: 2026-11-25  
File: `millions_of_paths_v2.py`

## Algorithm

- Run-length + DSU preprocessing labels 4-connected free-cell components.
- Disconnected endpoint pairs are answered immediately with `-1`.
- Two clear L-shaped paths are checked before graph search.
- Small bounding rectangles get an exact bitset-DP test for a Manhattan-length path.
- Hard queries use an exact search:
  - A* with Manhattan heuristic and deeper-node tie breaking when at least 0.5% of map cells are obstacles.
  - Bidirectional BFS on very sparse-obstacle maps.
- Repeated difficult pairs are cached.
- The padded grid removes neighbour-boundary branches in graph search.

A* remains exact: the Manhattan heuristic is admissible and consistent. The tie-break changes only the order of equal-priority states, not the returned shortest distance.

## Correctness checks

- Official sample 1: `5, 4, 0` — PASS.
- Official sample 2: `-1, 3, 0` — PASS.
- 20 random maps × 50 random queries, compared with reference BFS — PASS (1,000 queries).
- 15 sparse-obstacle maps × 30 random queries, compared with reference BFS — PASS (450 queries).
- Two 2048×2048 random-map queries cross-checked against a separate C++ BFS:
  - 70% free-cell map: distance 1336 — MATCH.
  - 80% free-cell map: distance 1910 — MATCH.

## Local performance checks

Environment: CPython 3.13.5, not the contest's Python 3.11/PyPy runtime. These are local measurements, not official judge results.

| Scenario | End-to-end time |
|---|---:|
| Open 2048×2048 map, 500,000 simple queries | 1.40 s |
| Split map, 500,000 disconnected queries | 1.18 s |
| Sparse vertical-wall map, one hard query | 2.51 s |
| Random 70%-free map, one hard query | 2.39 s |
| Random 80%-free map, one hard query | 2.12 s |

The one-query timings include map preprocessing. The A* search portion on the two random maps was approximately 0.30 s and 0.23 s, respectively.

## Important limitations

- The official 24-million-query test distribution is unknown.
- The obstacle-density threshold selecting A* versus bidirectional BFS is an empirical heuristic; other map structures may favour a different choice.
- No claim is made about the official score until this exact file is submitted to the judge.
- The code has been tested locally with Python 3.13.5; the contest's Python 3.11/PyPy runtime still needs a judge run.
