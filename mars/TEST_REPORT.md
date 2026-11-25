# Локальный отчёт — «Освоение Марса», агент v3

## 1. Почему появилась v3

Официальные результаты:
- v1: 6.24 (сообщённый пользователем результат, отображённый с округлением).
- v2: 6.236 (сообщённый пользователем результат).

v2 не улучшила результат. При аудите кода обнаружено, что она заменила более содержательную модель v1 — оценку вероятности конкуренции и будущей ценности расширения — упрощённым расстоянием до фронтира и грубым бонусом за соседей. Поэтому v3 создана от ветки v1, а не от v2.

## 2. Изменения v3

- Возвращена исходная модель v1: вероятность конкуренции зависит от расстояния до динамических фронтиров соперников, статического расстояния от станций и наличия доступных клеток у соперников.
- Возвращена оценка будущей ценности расширения через ещё не доступных соседей, с поправками на конкуренцию и геометрию.
- Параметр веса расширения изменён с 1.5 на 1.25.
- Вероятность конкуренции масштабируется коэффициентом 0.89.
- Сохранены fixed-point вычисления, heap и протокол v3.

## 3. Диагностическая настройка параметров

Проведён grid search на десяти открытых картах против исторических заявок соперников. Сравнивалась сумма конечного ресурса нашего игрока, делённого на число ходов карты. Для каждой карты применялось одно и то же правило: историческая заявка соперника обрезалась, если не помещалась в его фактический бюджет после изменения партии.

| Карта | v1: R/T | v3: R/T | Изменение |
|---:|---:|---:|---:|
| 01 | 4.210 | 4.250 | +0.040 |
| 02 | 5.663 | 5.796 | +0.133 |
| 03 | 2.723 | 3.223 | +0.500 |
| 04 | 6.175 | 6.585 | +0.410 |
| 05 | 12.000 | 13.420 | +1.420 |
| 06 | 15.671 | 15.404 | -0.267 |
| 07 | 0.330 | 0.330 | 0.000 |
| 08 | 4.466 | 4.466 | 0.000 |
| 09 | 7.895 | 7.895 | 0.000 |
| 10 | 554.589 | 554.701 | +0.111 |
| **Сумма** | **613.722** | **616.070** | **+2.348 (+0.38%)** |

В v3 на картах 04, 05 и 06 число ходов с обрезанной исторической заявкой соперника осталось тем же, что в v1: 2, 2 и 1 соответственно. На остальных картах обрезаний нет.

## 4. Ограничения интерпретации

Это не официальный score. Исторические соперники не адаптируют заявки к новому ходу партии; поэтому результат replay не гарантирует улучшения на закрытых картах или против скрытых стратегий. Кроме того, сумма R/T не тождественна формуле платформы, где используется нормировка и ограничение вклада отдельного теста.

## 5. Проверки

Ожидаемые CI-проверки на Python 3.11:
- компиляция и импорт;
- легальность заявок на всех десяти открытых картах;
- интерактивный протокол;
- точное воспроизведение десяти историй.

GitHub Actions, Python 3.11: **SUCCESS** — run [45657539518](https://github.com/Dmitriy1105/olimpiada/actions/runs/45657539518). Оба шага прошли: unit/interactive protocol tests и exact history validation.


## 6. Official platform result (user-reported)

The user submitted v3 and reported an official score of **6.122**.

Previously reported platform results:
- v1: **6.24** (reported to two decimal places).
- v2: **6.236**.
- v3: **6.122**.

Therefore v3 is rejected as a candidate for further submissions. Keep v1/v2 as the best known baseline; do not overwrite them.

The local replay gain reported in section 3 (+0.38% on the sum of (R/T)) did not transfer to the official score. This is consistent with the benchmark's known limitations: it covers only the 10 open easy maps, replays fixed historical opponent requests without running adaptive opponents, and clips requests when the historical action exceeds the changed budget. The package statement describes a 60-map evaluation including 50 hidden hard maps and opponents with 2–4 players. Thus the replay result was not a reliable predictor of official performance.

**Decision:** no more coefficient-only changes based on the current replay. The next experiment must improve evaluation coverage and test one policy change at a time. A candidate is accepted only after the user reports a better official score.
