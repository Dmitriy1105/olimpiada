# Task 1 — Жадные гномы

Интерактивная задача, Python 3.11+.

## Versions and official scores

- `solution_v11.py` — официальный результат **4.43**.
- `solution_v12.py` — текущий лучший официальный результат **4.676**.
- `solution_v13.py` — официальный результат **4.326**; эксперимент с адаптивным max/mean objective, ниже V12.
- `solution_v14.py` — эксперимент с более сильным обновлением модели соперников; официальный результат ожидается.

## V14 design

V14 is based on V12 and changes only the opponent-model update multipliers:
- predicted action matches observed action: `1.25 → 2.0`;
- predicted action differs: `0.85 → 0.5`.

All five opponent policies, four own continuation policies, 70/30 objective, pass handling, simulation logic, and tie-break terms are retained from V12.

## Tests and reports

- `V11_REPORT.md`, `V12_REPORT.md`, `V13_REPORT.md`, `V14_REPORT.md` — experiment reports.
- `bench_v14.py` — paired synthetic V12/V14 benchmark.
- `test_protocol_v14.py` — maximum-size interactive protocol test.

**Do not overwrite V12.** Keep it as the 4.676-point rollback version until an experiment receives a higher official score. Synthetic benchmark results are not official judge scores.
