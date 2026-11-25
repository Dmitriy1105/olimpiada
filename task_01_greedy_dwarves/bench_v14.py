"""Paired V12/V14 benchmarks with shared hidden opponent strategies."""
from __future__ import annotations

import importlib.util
import random
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V12_PATH = ROOT / "solution_v12.py"
V14_PATH = ROOT / "solution_v14.py"


def load_agent(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.GreedyDwarvesAgent


V12 = load_agent(V12_PATH, "v12_agent")
V14 = load_agent(V14_PATH, "v14_agent")


def random_instance(seed):
    rng = random.Random(seed)
    k = rng.randint(3, 10)
    n = rng.randint(10, 100)
    rounds = rng.randint(5, 20)
    capacity = rng.randint(10, 50)
    masses = [rng.randint(1, 50) for _ in range(n)]
    values = [rng.randint(1, 100) for _ in range(n)]
    return k, n, rounds, capacity, masses, values


def fixed_instance(seed):
    rng = random.Random(seed)
    masses = [rng.randint(1, 50) for _ in range(60)]
    values = [rng.randint(1, 100) for _ in range(60)]
    return 5, 60, 15, 35, masses, values


def hidden_orders(m, v):
    keys = [
        lambda i: v[i],
        lambda i: v[i] / m[i],
        lambda i: -m[i],
        lambda i: v[i] / (m[i] ** 0.5),
        lambda i: v[i] - 2 * m[i],
        lambda i: v[i] / (m[i] ** 0.7),
        lambda i: v[i] - 1.5 * m[i],
        lambda i: v[i] - 3 * m[i],
        lambda i: v[i] / (m[i] + 5),
        lambda i: v[i] / (m[i] + 10),
    ]
    return [sorted(range(len(m)), key=key, reverse=True) for key in keys]


def vmax_knapsack(m, v, capacity, rounds):
    neg = -10**9
    dp = [[neg] * (capacity + 1) for _ in range(rounds + 1)]
    dp[0][0] = 0
    for mass, value in zip(m, v):
        for count in range(rounds, 0, -1):
            for weight in range(capacity, mass - 1, -1):
                prev = dp[count - 1][weight - mass]
                if prev != neg:
                    dp[count][weight] = max(dp[count][weight], prev + value)
    return max(max(row) for row in dp)


def play(instance, hidden_order, agent_class):
    k, n, rounds, capacity, masses, values = instance
    agent = agent_class(k, n, rounds, capacity, masses, values)
    rank = {item: pos for pos, item in enumerate(hidden_order)}
    for _ in range(rounds):
        own_item = agent.choose()
        agent.commit_own_action(own_item)
        remaining = set(agent.available)
        local_masses = agent.masses.copy()
        actions = []
        for player in range(1, k):
            legal = [i for i in remaining if masses[i] <= capacity - local_masses[player]]
            if not legal:
                actions.append(0)
                continue
            item = min(legal, key=rank.get)
            remaining.remove(item)
            local_masses[player] += masses[item]
            actions.append(item + 1)
        agent.observe(actions)
    reward = agent.values[0] if agent.masses[0] < max(agent.masses[1:]) else 0
    vmax = vmax_knapsack(masses, values, capacity, rounds)
    return reward, reward / vmax if vmax else 0.0


def run_suite(name, seeds, instance_fn, hidden_fn):
    base, candidate = [], []
    start = time.perf_counter()
    for idx, seed in enumerate(seeds):
        instance = instance_fn(seed)
        _, _, _, _, masses, values = instance
        order = hidden_fn(idx, masses, values)
        base.append(play(instance, order, V12))
        candidate.append(play(instance, order, V14))
        if (idx + 1) % 50 == 0:
            print(f"{name}: {idx + 1}/{len(seeds)}", flush=True)
    wins = sum(y[1] > x[1] for x, y in zip(base, candidate))
    losses = sum(y[1] < x[1] for x, y in zip(base, candidate))
    ties = len(base) - wins - losses
    delta = statistics.mean(y[1] - x[1] for x, y in zip(base, candidate))
    print(f"\n{name}")
    for label, rows in (("V12", base), ("V14", candidate)):
        print(
            f"{label}: mean_reward={statistics.mean(x[0] for x in rows):.3f}, "
            f"mean_normalized={statistics.mean(x[1] for x in rows):.6f}, "
            f"positive={sum(x[0] > 0 for x in rows)}/{len(rows)}"
        )
    print(f"V14 vs V12: wins={wins}, losses={losses}, ties={ties}, mean_delta={delta:.7f}")
    print(f"elapsed_seconds={time.perf_counter() - start:.2f}")


def main():
    run_suite(
        "Randomized parameters, 300 games",
        range(63000, 63300),
        random_instance,
        lambda idx, m, v: hidden_orders(m, v)[idx % 5],
    )
    run_suite(
        "Fixed K=5,N=60,T=15,W=35, 300 games",
        range(64000, 64300),
        fixed_instance,
        lambda idx, m, v: hidden_orders(m, v)[idx % 5],
    )
    run_suite(
        "Expanded ten-policy pool, 200 games",
        range(67000, 67200),
        random_instance,
        lambda idx, m, v: hidden_orders(m, v)[idx % 10],
    )


if __name__ == "__main__":
    main()
