from statistics import mean

from simulator import make_instance, random_opponents, run_game

SEEDS = 60
K, N, T, W = 5, 60, 15, 35
modes = {
    "agent": ("agent", 0),
    "max_value": ("fixed", 0),
    "value_per_mass": ("fixed", 1),
    "value_per_sqrt_mass": ("fixed", 2),
    "value_minus_1.5m": ("fixed", 4),
    "min_mass": ("fixed", 6),
}
results = {name: [] for name in modes}
wins = {name: 0 for name in modes}

for seed in range(SEEDS):
    instance = make_instance(seed + 4000, K, N, T, W)
    opponents = random_opponents(seed + 8000, K)
    for name, (mode, policy) in modes.items():
        result = run_game(
            instance,
            opponents,
            own_mode=mode,
            own_policy=policy,
            scenario_count=6,
        )
        results[name].append(result.score)
        wins[name] += result.score > 0

for name in modes:
    print(f"{name:22s} avg_score={mean(results[name]):8.2f} positive={wins[name]:2d}/{SEEDS}")
