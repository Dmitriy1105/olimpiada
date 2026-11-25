# Millions of Paths — v3 verification report

Date: 2026-11-25  
Solver: `millions_of_paths_v3.py`  
Runtime used by CI: CPython 3.11.18

## Changes from v2

1. Fixed the vertical bitset conversion. The column array stores 0 for a free cell and 1 for an obstacle; the previous translation converted obstacles to set bits and free cells to zero bits. It now converts free cells to 1 and obstacles to 0, matching the row bitsets.
2. Use exact A* for every difficult connected query instead of selecting bidirectional BFS for low obstacle densities. The Manhattan heuristic remains admissible and consistent.
3. Use one direct `os.write(1, answer_bytes)` call per answer instead of two buffered writes plus an explicit flush.

## Python 3.11 CI

GitHub Actions run: `45644332836`  
Commit: `b15fa3f7ebe001612fc12b95473bded04b6aad43`  
Status: **SUCCESS**

Tests:
- Official sample 1: `5, 4, 0` — PASS.
- Official sample 2: `-1, 3, 0` — PASS.
- 15 random maps, all free-cell pairs compared with reference BFS — PASS.
- Explicit tall/narrow and wide/short bitset cases — PASS.
- Interactive request/response: parent sends one query and waits for one answer — PASS.
- Total unit tests in repository: 7 passed.

## Not established

- This CI run does not establish performance on 24 million queries.
- It does not establish a 20-second result on the official maximum-size maps.
- It does not establish the official score. Only the contest judge can do that.
- Earlier local timing claims should not be treated as a substitute for an end-to-end benchmark under the contest runtime.

## Next required measurement

Submit this exact file and record the official score and, if available, per-test accepted-query counts or runtime. If the score remains below 0.1, the next step is a workload-level profile: separate map preprocessing time, easy-query throughput, and hard-query search time. Further heuristic changes without those measurements would be guesswork.
