# Task 1 — Жадные гномы

Интерактивная задача, Python 3.11+.

## Versions

- solution_best_4_3.py — исходный пользовательский set-based алгоритм; ранее сообщался результат 4.30.
- solution_v11.py — ускоренная версия исходной пятистратегийной модели; официальный результат 4.43.
- solution_v12.py — добавлена четвёртая стратегия собственного продолжения; официальный результат 4.676.
- solution_v13.py — адаптивный коэффициент max/mean в зависимости от номера раунда; локально протестирована, официальный результат ожидается.
- test_v11.py — дифференциальные тесты V11 против set-based реализации.
- bench_v13.py — парное сравнение V12 и V13 на общих скрытых стратегиях.
- test_protocol_v13.py — subprocess-проверка интерактивного протокола на максимальных параметрах.
- V11_REPORT.md, V12_REPORT.md, V13_REPORT.md — отчёты версий.

## V13 change

V13 changes only the max/mean continuation blend:
- progress = round_no / max(1, T - 1)
- alpha = 0.85 - 0.30 * progress
- expected = alpha * max(results) + (1 - alpha) * mean(results)

Alpha decreases from 0.85 on the first round to 0.55 on the final round. The opponent model, own continuation policies, weight updates, pass handling, bit-mask simulation, and tie-break terms remain as in V12.

Keep V12 unchanged as the 4.676-point rollback baseline. Local synthetic tests are not official judge scores.
