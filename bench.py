"""Compare the adaptive agent with simple policies against one shared bot strategy."""
from statistics import mean

from simulator import make_instance, run_game

SEEDS = 20
K, N, T, W = 5, 60, 15, 35

# The hidden policy is selected once per test and is shared by every opponent.
# The candidate agent is evaluated on the same instances and opponents as baselines.
modes = {
    "adaptive_agent": ("agent", 0),
    "max_value": ("fixed", 0),
    "value_per_mass": ("fixed", 1),
    "value_per_sqrt_mass": ("fixed", 2),
    "linear_penalty": ("fixed", 4),
    "minimum_mass": ("fixed", 6),
    "maximum_mass": ("fixed", 7),
}
scores = {name: [] for name in modes}
positive = {name: 0 for name in modes}

for seed in range(SEEDS):
    instance = make_instance(seed + 30000, K, N, T, W)
    hidden_policy = seed % 10
    for name, (mode, own_policy) in modes.items():
        result = run_game(
            instance,
            hidden_policy,
            own_mode=mode,
            own_policy=own_policy,
        )
        scores[name].append(result.score)
        positive[name] += result.score > 0

for name in modes:
    print(
        f"{name:22s} "
        f"avg_score={mean(scores[name]):8.2f} "
        f"positive={positive[name]:2d}/{SEEDS}"
    )
