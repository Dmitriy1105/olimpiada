# Task 1 — Жадные гномы

Интерактивная задача, Python 3.11+.

## Versions

- solution_best_4_3.py — исходный пользовательский set-based алгоритм; ранее сообщался результат 4.3.
- solution_v11.py — оптимизация исходного пятистратегийного алгоритма без намеренного изменения решений.
- test_v11.py — дифференциальные тесты против буквальной set-based реализации.
- V11_REPORT.md — результаты проверок и локальный benchmark.
- solution_v5.py — отдельная экспериментальная версия с восемью моделями; не является базой V11.
- solution_v3.py and solution_v4.py — предыдущие эксперименты.

## V11 design

V11 preserves the five original opponent strategies, three own continuation policies, 70/30 objective, original tie-break terms, and 1.25/0.85 updates including opponent passes. It replaces repeated set/order scans inside simulations with rank bitmasks and capacity masks. The available set is retained for candidate iteration order.

Local differential tests are not official contest scores. V11 has not yet been submitted to the official judge.
