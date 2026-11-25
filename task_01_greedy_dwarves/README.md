# Task 1 — Жадные гномы

Интерактивная задача, Python 3.11+.

## Official-score history

- \`solution_v11.py\` — 4.43 (reported by the user).
- \`solution_v12.py\` — **4.676, current best official score**.
- \`solution_v13.py\` — 4.326 (reported by the user).
- \`solution_v14.py\` — approximately 4.3 (reported by the user).
- \`solution_v15.py\` — beam-search experiment; official score pending.

Never overwrite V12 until a new version receives a higher official score.

## V15: beam search with opponent rollouts

V15 keeps the five V12 opponent hypotheses and their online multiplicative weights. V12's four own greedy continuation policies are retained for rollouts, but the main decision now uses a two-round beam search over our future actions:
- shortlist up to 14 first actions using the V12 evaluator;
- branch over up to 6 diverse feasible future actions;
- simulate sequential responses from all opponents under each hidden-policy hypothesis;
- prune with value-maximizing and value-density rollouts;
- evaluate surviving states with the four V12 continuation policies.

Search settings: beam width 4, branch factor 6, depth 2.

## Validation files

- \`V11_REPORT.md\`, \`V12_REPORT.md\`, \`V13_REPORT.md\`, \`V14_REPORT.md\`, \`V15_REPORT.md\` — experiment records.
- \`bench_v15.py\` — reproducible paired synthetic V12/V15 comparison.
- \`test_protocol_v15.py\` — maximum-parameter interactive protocol test.

V15's local synthetic results are not official judge scores. V12 remains the rollback baseline until the official result is known.
