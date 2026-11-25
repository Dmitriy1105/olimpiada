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
