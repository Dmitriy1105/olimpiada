"""Paired V12/V13 simulation on deterministic shared-policy games."""
from __future__ import annotations

import importlib.util
import random
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_agent(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.GreedyDwarvesAgent


V12 = load_agent("solution_v12.py", "v12_agent")
V13 = load_agent("solution_v13.py", "v13_agent")


def random_instance(seed: int):
    rng = random.Random(seed)
    k = rng.randint(3, 10)
    n = rng.randint(10, 100)
    rounds = rng.randint(5, 20)
    capacity = rng.randint(10, 50)
    masses = [rng.randint(1, 50) for _ in range(n)]
    values = [rng.randint(1, 100) for _ in range(n)]
    return k, n, rounds, capacity, masses, values


def fixed_instance(seed: int):
    rng = random.Random(seed)
    k, n, rounds, capacity = 5, 60, 15, 35
    masses = [rng.randint(1, 50) for _ in range(n)]
    values = [rng.randint(1, 100) for _ in range(n)]
    return k, n, rounds, capacity, masses, values


def maximum_possible_value(masses, values, capacity, rounds):
    neg = -10**9
    dp = [[neg] * (capacity + 1) for _ in range(rounds + 1)]
    dp[0][0] = 0
    for mass, value in zip(masses, values):
        for count in range(rounds, 0, -1):
            for weight in range(capacity, mass - 1, -1):
                prev = dp[count - 1][weight - mass]
                if prev != neg:
                    dp[count][weight] = max(dp[count][weight], prev + value)
    return max(max(row) for row in dp)


def play(instance, hidden_policy, agent_class):
    k, n, rounds, capacity, masses, values = instance
    agent = agent_class(k, n, rounds, capacity, masses, values)
    for _ in range(rounds):
        own_item = agent.choose()
        agent.commit_own_action(own_item)

        # All opponents share the same hidden policy and act sequentially.
        bot_mask = agent.remaining_masks[hidden_policy]
        actions = []
        for player in range(1, k):
            item = agent._pick(
                hidden_policy,
                bot_mask,
                capacity - agent.masses[player],
            )
            if item >= 0:
                bot_mask &= ~agent.item_bits[hidden_policy][item]
                actions.append(item + 1)
            else:
                actions.append(0)
        agent.observe(actions)

    reward = (
        agent.values[0]
        if agent.masses[0] < max(agent.masses[1:])
        else 0
    )
    vmax = maximum_possible_value(masses, values, capacity, rounds)
    return reward, reward / vmax if vmax else 0.0


def summarize(name, baseline, candidate):
    print(f"\n{name}")
    for label, rows in (("V12", baseline), ("V13", candidate)):
        print(
            f"{label}: mean_reward={statistics.mean(x[0] for x in rows):.3f}, "
            f"mean_normalized={statistics.mean(x[1] for x in rows):.6f}, "
            f"positive={sum(x[0] > 0 for x in rows)}/{len(rows)}"
        )
    wins = sum(y[1] > x[1] for x, y in zip(baseline, candidate))
    losses = sum(y[1] < x[1] for x, y in zip(baseline, candidate))
    ties = len(baseline) - wins - losses
    delta = statistics.mean(y[1] - x[1] for x, y in zip(baseline, candidate))
    print(
        f"paired V13 vs V12: wins={wins}, losses={losses}, "
        f"ties={ties}, mean_delta={delta:.7f}"
    )


def main():
    start = time.perf_counter()

    random_base, random_new = [], []
    for idx in range(120):
        seed = 41000 + idx
        instance = random_instance(seed)
        hidden = random.Random(seed + 999).randrange(5)
        random_base.append(play(instance, hidden, V12))
        random_new.append(play(instance, hidden, V13))
    summarize("Randomized parameters (120 games)", random_base, random_new)

    fixed_base, fixed_new = [], []
    for idx in range(150):
        instance = fixed_instance(41000 + idx)
        hidden = idx % 5
        fixed_base.append(play(instance, hidden, V12))
        fixed_new.append(play(instance, hidden, V13))
    summarize(
        "Balanced shared-policy set (150 games; K=5,N=60,T=15,W=35)",
        fixed_base,
        fixed_new,
    )

    print(f"\nElapsed seconds: {time.perf_counter() - start:.2f}")


if __name__ == "__main__":
    main()
