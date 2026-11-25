# Задача 7 — «Последний метр»: экспериментальный отчёт

## Источник и формат

Проверен архив участника `last-metre-participant-v2.zip`. Архитектура сети: `42 → 32 → 32 → 9`, `Tanh` после скрытых слоёв. Штатный загрузчик требует исходные имена и формы state_dict, конечные значения float32.

## Кандидат

`weights.pt` построен от `examples/weights.pt`:
- первая строка весов первого Linear-слоя умножена на 1.6;
- вторая строка весов первого Linear-слоя умножена на 1.3;
- остальные параметры не менялись.

Файл проверен штатным `torch_lander.load`:
- `0.weight`: (32, 42), float32, finite;
- `0.bias`: (32,), float32, finite;
- `2.weight`: (32, 32), float32, finite;
- `2.bias`: (32,), float32, finite;
- `4.weight`: (9, 32), float32, finite;
- `4.bias`: (9,), float32, finite.

## Парные тесты

В каждой паре baseline и кандидат запускались на одних и тех же шести эпизодах `sample(6, seed, profile='medium')`, с `score_mode='contest'`, `horizon=800`, детерминированным `argmax`.

| Seed | Baseline R | Кандидат R | Baseline посадок | Кандидат посадок |
|---:|---:|---:|---:|---:|
| 80021 | 333.43 | 334.23 | 2/6 | 2/6 |
| 90073 | 275.41 | 478.90 | 2/6 | 3/6 |
| 12345 | -62.67 | 139.77 | 0/6 | 1/6 |
| 23456 | -4.17 | 20.14 | 1/6 | 1/6 |
| **Итого** | **542.00** | **973.04** | **5/24** | **7/24** |

На этих 24 локальных эпизодах суммарное вознаграждение кандидата выше на 431.05. Результат неодинаков по seed; это небольшой экспериментальный набор, а не гарантия роста на официальных 240 фиксированных полётах.

## Ограничения

- Полная официальная проверка на 240 полётах локально не запускалась.
- Локальная среда имеет PyTorch 2.10.0 CPU, тогда как в условии архива указано PyTorch 2.2.2; файл совместим по формату, но окончательное решение принимает официальный рантайм.
- В присланных скриншотах формула предварительного балла содержит множитель 12, а в `statement-russian.md` и `torch_lander/score_curve.py` архива v2 задан максимум 10. При оптимизации весов это не влияет на сравнение, так как оптимизируется сырое вознаграждение R. Для финального зачёта следует ориентироваться на фактический чекер соревнования.

## Вывод

Кандидат перспективнее стартовой модели на небольшом локальном наборе. Официальная отправка прошла: пользователь сообщил предварительный балл **1,70 из 12** (25.11.2026). Сохранить v1 как базу для сравнения будущих моделей. Сырые результаты отдельных закрытых полётов недоступны.


## CELL 07 — локальная диагностика (25.11.2026)

Источник: результаты запуска пользователем в Jupyter Notebook. `sample(60, 73291, profile='medium')`, `score_mode='contest'`, horizon=800, PyTorch 2.11.0+cu128, один CPU-поток, по 10 полётов на семейство.

| Family | Starter landed | v1 landed | Starter R | v1 R | Starter hull contacts | v1 hull contacts |
|---|---:|---:|---:|---:|---:|---:|
| approach | 0 | 1 | -245.17 | -38.47 | 0 | 0 |
| lateral | 2 | 4 | 291.56 | 622.87 | 0 | 0 |
| fast_descent | 4 | 4 | 816.81 | 853.25 | 0 | 0 |
| recovery | 1 | 1 | 183.33 | 195.00 | 1 | 1 |
| low_thrust | 5 | 7 | 610.22 | 1013.75 | 0 | 0 |
| crosswind | 0 | 0 | -564.74 | -625.17 | 2 | 3 |
| Total | 12/60 | 17/60 | 1092.03 | 2021.22 | 3 | 4 |

Время 114.54 / 115.04 сек. Суммарные награды взяты из прямого вывода, значения по семействам округлены. Выборку теперь считать диагностической, не независимой финальной валидацией. v1 остаётся базой (официально 1.70/12). Следующий этап — контракт признаков и диагностика причин завершения эпизодов, затем новый эксперимент обучения. Не считать рост локального R прогнозом официального балла.


## CELL 08 — Observation & Action Contract

QC PASS по выводу пользователя: 60 начальных наблюдений, shape (60, 42), float32, 9 действий, все значения конечны. Контракт сверен с `torch_lander/env.py` из исходного архива. Признаки 25–26 равны нулю по конструкции площадки; признаки 35–41 описывают изменяемое состояние опор и время стабилизации, одинаковые при инициализации. Нулевая дисперсия на начальных наблюдениях не является основанием для их удаления. Входы 10–11 кодируют отдельно мощности боковых двигателей, 12 — нормализованный ветер. CELL 08A: результаты анализа весов ещё не получены.


## CELL 08A — Control Channel Analysis

По выводу пользователя: первый управляющий канал v1 использует height=-0.128, vy=-0.160, контакты опор=-0.320 каждый; второй relative_x=0.0364, vx=0.0637, sin_theta=-0.1300, omega=-0.0780. В обоих прямые веса признаков 8–14 равны нулю. Остальные нейроны имеют ненулевые веса, поэтому это не доказательство отсутствия влияния этих признаков на всю сеть. Выходные коэффициенты первого канала [0,0,0,0,0,0,10,10,10], второго [-10,0,10,-10,0,10,-10,0,10]. Вывод последней проверки числа параметров не прислан. Следующий этап CELL 09: детальная диагностика финальных состояний и частот действий на той же выборке 60 полётов (seed 73291). Это диагностическая, не независимая выборка.


## CELL 09 — Flight Failure Analysis

QC PASS по результатам пользователя (60 полётов, seed 73291, horizon 800): R=2021.22, LANDED=17, TIMEOUT=26, INVALID_LANDING=13, HULL_CONTACT=4, всего 36467 действий, 117.88 сек. Средние финальные показатели TIMEOUT: |dx|=0.504, y=0.346, vy=0.022. Crosswind: 0 посадок, 7 INVALID_LANDING, 3 HULL_CONTACT. Действия 3–5 (50% тяги) не использовались. Следующая гипотеза: отдельный эвристический учитель, учитывающий ветер, гравитацию, доступную тягу и мощности поворота, с целью уменьшить зависание. Экспериментальный код учителя в controller_v2.py; результат CELL 10 пока неизвестен. Учитель не является допустимым файлом для отправки; при успешной диагностике необходима дистилляция в фиксированную сеть.


## CELL 10B — Teacher evaluation

Результаты пользователя на диагностических 60 полётах: R=884.86 (v1=2021.22, delta=-1136.36), посадки 10/60, TIMEOUT=50, HULL_CONTACT=0. Low_thrust 9/10, crosswind 0/10. Половинная тяга 81.357% действий. QC формата PASS, эксперимент NOT IMPROVED. Не использовать учителя как источник меток. Следующий эксперимент CELL 10C: контролируемая абляция смещения первого скрытого нейрона v1 (постоянная добавка к порогу вертикального канала), предварительный скрининг 18 эпизодов; подтверждение на 60 и независимая проверка. Изменение не считать улучшением до измерения R. v1 сохранена.


## CELL 10C — Descent channel optimization

Пользовательский запуск: 18 полётов, seed 48371, medium, horizon 800. Изменяется только 0.bias[0] относительно v1.

| Candidate | Bias | R | Landed | Timeout | Hull | Mean steps |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 0 | 686.4512 | 5 | 9 | 0 | 608.6667 |
| v2a | -0.0064 | 1452.4005 | 10 | 0 | 1 | 285.2222 |
| v2b | -0.0128 | 1472.0182 | 10 | 0 | 0 | 227 |
| v2c | -0.0192 | 1688.0044 | 11 | 0 | 0 | 195.8889 |

QC PASS, экспорт и повторная загрузка кандидатов прошли по выводу пользователя. v2c предварительный победитель, не подтверждён официальным чекером. Файлы сохранены локально у пользователя, не загружены в репозиторий. Следующий эксперимент CELL 10D: проверить v2c на прежних 60 диагностических полётах seed 73291, затем независимая парная проверка v1 и v2c на новом seed; исходные веса не перезаписывать.


## CELL 10D — V2C diagnostic validation

Пользовательский запуск на прежней диагностической выборке 60 полётов, seed 73291, medium, horizon 800. R=4534.722 (v1=2021.22, +124.36%), LANDED=33/60, TIMEOUT=0, HULL_CONTACT=7, runtime=28.69 sec. По семействам посадки: approach 8, lateral 4, fast_descent 9, recovery 6, low_thrust 6, crosswind 0 (по 10). QC PASS, EXPERIMENT IMPROVED. Ухудшение по hull contacts (4→7), low_thrust (7→6), lateral R (622.87→465.573). v2c — текущий локальный лидер, официальный балл пока неизвестен. Следующий этап CELL 10E: парное сравнение v1 и v2c на новом seed 95173, 60 полётов; после использования для выбора модели эту выборку не считать финальной независимой проверкой. Бинарные файлы остаются локально у пользователя.


## CELL 10E — Holdout validation

Пользовательский парный прогон на новой выборке 60 полётов (seed 95173, medium, horizon 800). v1: R=3068.522, landed=22, timeout=24, hull=5, mean steps=598.20. v2c: R=5428.768, landed=37, timeout=0, hull=4, mean steps=195.95. Delta R=2360.246 (+76.92%), 43 выигрыша / 17 проигрышей / 0 ничьих. Средний парный прирост=39.337, приближённый 95% bootstrap CI=[6.541,72.167] (обычный непараметрический bootstrap из ячейки, не стратифицированный). QC PASS, EXPERIMENT IMPROVEMENT SUPPORTED. Посадки v1→v2c по семействам: approach 3→10, lateral 4→7, fast_descent 5→9, recovery 0→7, low_thrust 10→4, crosswind 0→0. v2c — локальный лидер; официальный балл пока неизвестен. Выборка 95173 теперь использована для оценки и диагностики; финальный тест должен использовать новые seed. Следующие направления: исправление регрессии low_thrust, оценка горизонтального канала на отдельных development seeds. Не заменять v2c до сравнительной проверки.


## CELL 11A — Planned adaptive channel screening

Base v2c; one auxiliary hidden neuron selected by smallest absolute path-weight proxy, replaced with tanh gate based on visible normalized gravity and thrust. Threshold thrust/g=1.675 follows public medium-profile ranges, not hidden flights. Correction to second hidden preactivation: alpha*(1+gate)/2. Candidates alpha=0 (ablation), 0.0096, 0.0192, 0.0288 plus unchanged v2c; new development seed 61429, 24 flights. Local shape/finite smoke test of equivalent candidate passed (24×9, 2729 parameters); actual performance not yet measured. This is an approximation, not exact switching between v1/v2c. Validate on fresh seeds before promotion. Code: adaptive_v3.py.


## CELL 11A.2 — Adaptive model screening

Пользовательский запуск на development seed 61429, 24 полёта (4 на семейство), medium, horizon 800. Результаты: v2c R=1471.778, landed=11, timeout=0, hull=2; v3_control R=1472.488, landed=11; v3a R=1671.545, landed=12; v3b R=1859.718, landed=13, timeout=0, hull=2; v3c R=1368.117, landed=10, timeout=4, hull=2. v3b прирост +387.940 (+26.36%) относительно v2c. По low_thrust: v2c 1/4 (R=57.505), v3a 2/4 (R=257.254), v3b 3/4 (R=445.427), v3c 0/4 (R=-46.174). Остальные семейства имеют одинаковое число посадок между кандидатами; crosswind 0/4. QC PASS. v3b — предварительный кандидат, не promoted: sample всего 24, подбор alpha выполнялся на этом seed. Следующий шаг CELL 11A.3 — paired validation v2c vs v3b на новом seed 81739, 60 эпизодов (10 на семейство), стратифицированный bootstrap CI по семействам. Не использовать seed 61429 повторно для подтверждения.


## CELL 11A.3 — Paired validation v2c vs v3b

Пользовательский запуск: новый seed 81739, 60 парных полётов (10 на семейство), medium, horizon 800. v2c: R=5470.125, landed=37/60 (61.667%), timeout=0, hull=8, mean steps=198.817. v3b: R=5871.526, landed=39/60 (65.0%), timeout=0, hull=8, mean steps=242.450. Delta total R=+401.401 (+7.34%), +2 посадки. Парные outcomes: 13 wins / 10 losses / 37 ties; mean paired delta=6.690. Stratified bootstrap 95% CI=[-6.849,19.836], includes zero. QC PASS; IMPROVED BUT UNCERTAIN. v2c remains incumbent; v3b candidate only. Seed 81739 is now consumed and must not be reused for tuning or final validation. Next CELL 11A.4: expanded paired comparison on a fresh seed, 240 episodes (40 per family), seed 62814, stratified paired bootstrap; compare only v2c and v3b to avoid further selection on a small sample.


## CELL 11A.4 — Expanded paired validation v2c vs v3b

Пользовательский запуск на seed 62814, 240 полётов (40 на каждое из 6 семейств), medium, horizon 800. v2c: R=19432.072, landed=135/240 (56.25%), timeout=0, hull=17, mean steps=195.883. v3b: R=21943.638, landed=148/240 (61.667%), timeout=0, hull=16, mean steps=244.742. Delta R=+2511.566 (+12.93%), +13 посадок (+5.417 п.п.), hull contacts -1, timeout unchanged at zero. Paired outcomes: 46 wins / 69 losses / 125 ties; mean paired delta=10.465 per episode. Stratified bootstrap 95% CI for mean family-weighted delta=[4.429,16.637]; paired total-reward CI=[897.894,4299.615]. Both intervals exclude zero. QC PASS; IMPROVEMENT SUPPORTED. Promote v3b to current experimental incumbent; retain v2c as rollback. Do not overwrite original weights. Next step: inspect CELL 11A.4 family-level CSV to identify weak family; prioritize crosswind diagnosis only if family results confirm it remains the dominant failure. Seed 62814 is consumed and not for future tuning/final test.


## CELL 11B.1 — Family performance interpretation and diagnostic plan

From CELL 11A.4 family table (40 episodes/family): v3b vs v2c reward deltas: approach -0.453 (landings 31→31); crosswind +4.611 (1→1 landing, 11→11 hull contacts); fast_descent -0.231 (37→37); lateral +199.350 (21→22); low_thrust +2515.441 (22→35, hull 1→0); recovery -207.153 (23→22). Thus nearly all aggregate v3b improvement comes from low_thrust; crosswind remains severe and unchanged, recovery has a modest regression. Next step is diagnostics, not blind parameter edits: compare v2c/v3b per-episode initial/final observation features and action distributions on a fresh development seed 73129, 240 medium episodes (40/family). Script task7/cell11b1_crosswind_diagnostics.py records all 42 observation features and per-episode action counts; outputs CSVs. Seed is development only and must not be presented as final holdout.


## CELL 11B.1 — Crosswind/recovery diagnostic results (development seed 73129)

Пользовательский запуск на общей парной выборке 240 эпизодов (40 на семейство), medium, horizon 800. v2c: R=19680.87, landed=137/240, timeout=0, hull=20. v3b: R=21197.18, landed=145/240, timeout=0, hull=19. Разница v3b−v2c: R=+1516.31, посадки +8, hull −1. Это development-диагностика, не независимый финальный holdout.

По семействам (v3b−v2c): approach R +0.515, посадки без изменения; crosswind R −44.162, посадки 0→0, hull 9→12; fast_descent R −1.054, посадки без изменения; lateral R −175.634, посадки 23→22; low_thrust R +1736.375, посадки 20→29, hull 3→0; recovery R +0.272, посадки без изменения. Прирост в основном обеспечен low_thrust, а crosswind и lateral требуют отдельной диагностики.

Crosswind action distribution практически совпадает: действия 3–5 (50% throttle) не выбирались обеими моделями. Доля нулевой тяги: v2c 61.646%, v3b 61.713%; полной тяги: 38.355% и 38.286%. Направления поворота: left 16.210%→15.469%, center 67.897%→67.123%, right 15.894%→17.407%. Это не доказывает причину отказов, но показывает, что адаптация v3b почти не меняет горизонтальное поведение.

**Исправление отчётности:** первоначальный CELL 11B.1 помечал все эпизоды, не являющиеся LANDED/HULL_CONTACT/TIMEOUT, как OTHER. Это упрощение недостаточно для анализа. Скрипт обновлён: теперь статус вычисляется теми же правилами и в том же порядке, что CELL 09 — LANDED, HULL_CONTACT, TIMEOUT, OUT_OF_BOUNDS, INVALID_LANDING (settle_time >= 0.5−1e−10), OTHER. Добавлены final_x, final_y и final_settle_time. Повторный запуск на seed 73129 нужен только для корректной классификации тех же эпизодов; это не новая выборка. Все initial_*/final_* признаки в CSV — нормализованные входы модели, не физические единицы.

Следующий шаг — повторить CELL 11B.1 на том же seed для получения корректного terminal-status breakdown. Только после этого выбрать ограниченный горизонтальный канал эксперимент; v3b сохранить как текущего экспериментального лидера, v2c — как rollback.


### CELL 11B.1R — correction after runtime API mismatch

Первый повторный запуск остановился после завершения прогона v2c с `AttributeError: 'Lander' object has no attribute 'x'`. Причина — диагностический скрипт обращался к несуществующим `env.x` и `env.y`. Проверен исходный `torch_lander/env.py` из `last-metre-participant-v2.zip`: физическое состояние доступно через `env.state` (колонки `x`, nominal height, `vx`, `vy`, `theta`, `omega`); `env.settle_time` и `env.terminated` входят в публичный набор dynamic_fields. Скрипт исправлен и теперь использует эти реальные поля. Классификация следует точной логике исходного `env.py`: `terminated = crash | outside | settled`; `hull_contact` выделяет crash, `settle_time >= 0.5−1e−10` — settled. Поэтому `terminated & settled & ~landed & ~hull_contact` → INVALID_LANDING, а `terminated & ~settled & ~hull_contact` → OUT_OF_BOUNDS. TIMEOUT, HULL_CONTACT и LANDED берутся из соответствующих публичных флагов. Повторный запуск CELL 11B.1R выполнен на том же development seed 73129: v2c R=19680.87, landed=137/240, hull=20; v3b R=21197.18, landed=145/240, hull=19. Исправленная классификация: crosswind v2c — 0 landed / 31 INVALID_LANDING / 9 HULL_CONTACT; v3b — 0 / 28 / 12. См. следующий раздел CELL 11B.1R.


## CELL 11B.1R — corrected terminal status and pad-offset analysis

Пользовательский повторный запуск после исправления API, тот же development seed 73129, 240 эпизодов (40 на семейство), horizon 800. QC PASS.

| Model | R | Landed | Hull contacts | Timeouts |
|---|---:|---:|---:|---:|
| v2c | 19680.87 | 137/240 | 20 | 0 |
| v3b | 21197.18 | 145/240 | 19 | 0 |

По семействам: approach 30→30; crosswind 0→0; fast_descent 33→33; lateral 23→22; low_thrust 20→29; recovery 31→31. Изменение общего результата обеспечено главным образом low_thrust (+9 посадок). Crosswind остаётся нерешённым.

**Корректная терминальная классификация crosswind (40 эпизодов на модель):**

| Status | v2c | v3b |
|---|---:|---:|
| LANDED | 0 | 0 |
| INVALID_LANDING | 31 | 28 |
| HULL_CONTACT | 9 | 12 |
| TIMEOUT | 0 | 0 |

По исходному env.py INVALID_LANDING означает устойчивую двухопорную поддержку и выдерживание SETTLE_SECONDS, но условие on_pad не выполнено. Для on_pad проверяются геометрическое положение обеих опор в пределах площадки, высота стоп и угол корпуса. Следовательно, это не просто «неустойчивое касание».

Из финального observation: relative_x = (x - pad_center)/20. Поэтому |final_relative_x|×20 даёт абсолютное смещение корпуса от центра площадки в координатах симулятора. Среднее смещение по crosswind failure rows:
- v2c INVALID_LANDING: 10.220; HULL_CONTACT: 12.347.
- v3b INVALID_LANDING: 10.568; HULL_CONTACT: 10.340.
Медианное абсолютное смещение по всем 40 эпизодам: v2c 10.83, v3b 10.27. Только 1/40 эпизодов v2c и 2/40 v3b оказались в пределах максимальной ширины площадки 2.4. Это указывает на крупную ошибку горизонтального позиционирования; финальная классификация сама по себе не доказывает, что только она является причиной каждого invalid landing.

Crosswind action shares:
- v2c: actions 0..8 = 15.246, 31.561, 14.839, 0, 0, 0, 0.964, 36.336, 1.055%.
- v3b: actions 0..8 = 14.277, 31.177, 16.259, 0, 0, 0, 1.192, 35.946, 1.148%.
Action mapping: throttle=(action//3)*0.5; turn=action%3−1. Thus both policies use only 0% and 100% throttle in this family; actions 3–5 (50% throttle) are unused. Their overall crosswind action distributions are very similar.

The final feature table is a diagnostic subset of all crosswind/recovery rows; the full per-episode diagnostics and trajectory/action CSVs remain in the user's REPORTS_DIR. Paired wins/losses/ties and confidence intervals were not present in the supplied console output and are not inferred here. Seed 73129 remains development-only, not a final holdout.

## CELL 11B.2 — signed-wind horizontal-channel screening (prepared)

Hypothesis: v3b's adaptive gate changes the vertical/thrust channel; the crosswind action distribution is nearly unchanged. Test a small signed-wind feed-forward term in the existing horizontal turn channel while preserving the v3b vertical gate and original weights.

Script: task7/cell11b2_horizontal_wind_screen.py. It identifies the existing v3b vertical gate, selects a low-influence auxiliary first-layer neuron, and routes normalized signed wind observation feature 12 into the verified turn-channel neuron 1. Four signed amplitudes are screened: −0.0192, −0.0096, +0.0096, +0.0192; gate sharpness=2.0. Positive and negative directions are both tested rather than assumed.

Protocol: compare baseline v3b and four candidates on the same 48 crosswind-only medium episodes, seed 93517, horizon 800. Save candidate weights separately and export episode, trajectory, action, paired-screen, and candidate-manifest CSVs. This is a small development screen only; no candidate is promoted from this result alone. Any promising candidate must next be checked on a fresh balanced 240-episode run across all six families, with paired and stratified-bootstrap analysis. Original v2c and v3b weights are not overwritten.


### CELL 11B.2 — implementation correction

Первый запуск остановился до генерации кандидатов с `RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation` при обнулении строки весов вспомогательного нейрона. Это ошибка режима autograd во время структурной модификации параметров, а не ошибка архитектуры или данных. В `_make_wind_candidate()` все in-place изменения весов и bias обёрнуты в `torch.no_grad()`. Исправление сохранено в GitHub (commit `4b50e47`). Повторный запуск CELL 11B.2 ещё не выполнен; никаких результатов screening пока нет.


## CELL 11B.2 — signed-wind horizontal-channel screening results

Пользовательский запуск на development seed 93517, 48 crosswind episodes per model, medium, horizon 800. QC PASS. Baseline v3b: R=-2464.952, landed=0/48, hull=14, timeout=0, mean absolute final pad offset=11.971, median=11.164, mean closest offset below height 5=9.898.

Candidates:
- alpha -0.0192: R=-3022.749, landed=0, hull=16, timeout=0, mean abs final offset=17.691.
- alpha -0.0096: R=-2761.579, landed=0, hull=17, timeout=1, mean abs final offset=15.004.
- alpha +0.0096: R=-1995.822, landed=0, hull=13, timeout=1, mean abs final offset=7.209.
- alpha +0.0192: R=-1431.959, landed=1, hull=5, timeout=1, mean abs final offset=5.005, median=4.835, mean closest offset below height 5=3.665.

Paired against v3b:
- alpha -0.0192: 7 wins / 41 losses, mean delta=-11.6208, bootstrap 95% CI [-15.3330,-7.9678], hull delta +2.
- alpha -0.0096: 13/35, mean delta=-6.1797, CI [-11.0325,-1.3673], hull delta +3.
- alpha +0.0096: 38/10, mean delta=+9.7736, CI [6.2197,13.4929], hull delta -1.
- alpha +0.0192: 44/4, mean delta=+21.5207, CI [13.4936,34.1320], landing delta +1, hull delta -9.

Best preliminary candidate is `v3b_wind_p00192` (alpha +0.0192). Compared with v3b, its total reward is +1032.993 higher (41.9% reduction in the magnitude of the negative reward), hull contacts fall from 14 to 5 (-64.3%), mean final absolute pad offset falls from 11.971 to 5.005 (-58.2%), and it lands 1/48 rather than 0/48. It also has one timeout vs zero for v3b. Strong development-screen signal only: a single 48-episode crosswind seed is not sufficient for promotion or final claims. The 95% paired bootstrap interval is conditional on this screening sample and does not correct for candidate selection.

The printed `invalid_landings=0` is only one terminal-status column; CELL 11B.2 did not print OUT_OF_BOUNDS/OTHER counts. Do not infer that every non-landed episode was a hull contact or that all terminal statuses are explained. CELL 11B.3 adds a complete terminal-status breakdown.

## CELL 11B.3 — full paired validation prepared

Compare incumbent `weights_v3b.pt` with preliminary winner `weights_v3b_wind_p00192.pt` on a fresh balanced 240-episode batch (40 per family), medium, horizon 800, seed 94627. This seed is distinct from CELL 11A.4 (62814), CELL 11B.1R (73129), and CELL 11B.2 (93517). No tuning occurs in this stage.

Script: `task7/cell11b3_full_validation.py`. Outputs flights, overall summary, family summary, paired results, stratified family-weighted bootstrap CI, total-reward paired bootstrap CI, and terminal status counts (LANDED, HULL_CONTACT, TIMEOUT, INVALID_LANDING, OUT_OF_BOUNDS, OTHER). Original v3b/v2c weights are not overwritten. Candidate is not promoted until aggregate evidence and family-level regressions are reviewed.
