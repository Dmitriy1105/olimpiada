"""Reproducible paired synthetic benchmark: V12 vs V15."""
import importlib.util
import random
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_agent(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GreedyDwarvesAgent


V12 = load_agent("solution_v12.py", "v12_agent")
V15 = load_agent("solution_v15.py", "v15_agent")


def make_instance(seed, fixed=False):
    rng = random.Random(seed)
    if fixed:
        k, n, rounds, capacity = 5, 60, 15, 35
    else:
        k = rng.randint(3, 10)
        n = rng.randint(10, 100)
        rounds = rng.randint(5, 20)
        capacity = rng.randint(10, 50)
    masses = [rng.randint(1, 50) for _ in range(n)]
    values = [rng.randint(1, 100) for _ in range(n)]
    return k, n, rounds, capacity, masses, values


def hidden_orders(masses, values, expanded=False):
    keys = [
        lambda i: values[i],
        lambda i: values[i] / masses[i],
        lambda i: -masses[i],
        lambda i: values[i] / (masses[i] ** 0.5),
        lambda i: values[i] - 2 * masses[i],
    ]
    if expanded:
        keys += [
            lambda i: values[i] / (masses[i] ** 0.7),
            lambda i: values[i] - 1.5 * masses[i],
            lambda i: values[i] - 3 * masses[i],
            lambda i: values[i] / (masses[i] + 5),
            lambda i: values[i] / (masses[i] + 10),
        ]
    return [
        sorted(range(len(masses)), key=key, reverse=True)
        for key in keys
    ]


def maximum_value(masses, values, capacity, rounds):
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


def play(instance, hidden_order, agent_class):
    k, n, rounds, capacity, masses, values = instance
    agent = agent_class(k, n, rounds, capacity, masses, values)
    available = set(range(n))
    player_masses = [0] * k
    player_values = [0] * k

    for _ in range(rounds):
        own_item = agent.choose()
        agent.commit_own_action(own_item)
        if own_item >= 0:
            available.remove(own_item)
            player_masses[0] += masses[own_item]
            player_values[0] += values[own_item]

        actions = []
        for player in range(1, k):
            item = -1
            for candidate in hidden_order:
                if candidate in available and (
                    player_masses[player] + masses[candidate] <= capacity
                ):
                    item = candidate
                    break
            if item >= 0:
                available.remove(item)
                player_masses[player] += masses[item]
                player_values[player] += values[item]
                actions.append(item + 1)
            else:
                actions.append(0)
        agent.observe(actions)

    reward = (
        player_values[0]
        if player_masses[0] < max(player_masses[1:])
        else 0
    )
    vmax = maximum_value(masses, values, capacity, rounds)
    return reward, reward / vmax if vmax else 0.0


def summarize(title, rows):
    v12 = [x[0] for x in rows]
    v15 = [x[1] for x in rows]
    wins = sum(b[1] > a[1] for a, b in rows)
    losses = sum(b[1] < a[1] for a, b in rows)
    ties = len(rows) - wins - losses
    print(f"\n{title} — {len(rows)} paired games")
    for name, data in (("V12", v12), ("V15", v15)):
        print(
            f"{name}: mean_reward={statistics.mean(x[0] for x in data):.3f}, "
            f"mean_normalized={statistics.mean(x[1] for x in data):.6f}, "
            f"positive={sum(x[0] > 0 for x in data)}/{len(data)}"
        )
    print(
        f"V15 paired: wins={wins}, losses={losses}, ties={ties}, "
        f"mean_delta={statistics.mean(b[1] - a[1] for a, b in rows):.7f}"
    )


def run_suite(seeds, fixed, expanded, title):
    rows = []
    for idx, seed in enumerate(seeds):
        instance = make_instance(seed, fixed)
        _, _, _, _, masses, values = instance
        orders = hidden_orders(masses, values, expanded)
        hidden = idx % len(orders)
        rows.append((
            play(instance, orders[hidden], V12),
            play(instance, orders[hidden], V15),
        ))
        if (idx + 1) % 25 == 0:
            print(f"{title}: {idx + 1}/{len(seeds)}", flush=True)
    summarize(title, rows)


def main():
    start = time.perf_counter()
    run_suite(range(81000, 81120), False, False, "Random parameters; 5 hidden policies")
    run_suite(range(82000, 82100), True, False, "Fixed K=5,N=60,T=15,W=35; 5 hidden policies")
    run_suite(range(83000, 83100), True, True, "Fixed parameters; expanded 10-policy pool")
    print(f"\nTotal elapsed seconds: {time.perf_counter() - start:.2f}")


if __name__ == "__main__":
    main()
