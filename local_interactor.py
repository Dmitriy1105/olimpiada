"""Subprocess-level interactive protocol test using one shared bot strategy."""
from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

from solution import GreedyDwarvesAgent


def run_protocol(seed: int = 77, k: int = 10, n: int = 100, rounds: int = 20, capacity: int = 50):
    rng = random.Random(seed)
    masses = [rng.randint(1, 50) for _ in range(n)]
    values = [rng.randint(1, 100) for _ in range(n)]
    hidden_policy = rng.randrange(10)

    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("solution.py"))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None and process.stdout is not None

    process.stdin.write(f"{k} {n} {rounds} {capacity}\n")
    process.stdin.write(" ".join(map(str, masses)) + "\n")
    process.stdin.write(" ".join(map(str, values)) + "\n")
    process.stdin.flush()

    available = set(range(n))
    player_mass = [0] * k
    player_value = [0] * k
    model = GreedyDwarvesAgent(k, n, rounds, capacity, masses, values, cpu_budget=100)
    order = model.orders[hidden_policy]
    outputs = []

    for _ in range(rounds):
        line = process.stdout.readline()
        if not line:
            raise AssertionError("solver exited before producing all T actions")
        action = int(line.strip())
        outputs.append(action)

        if action == 0:
            if any(masses[i] <= capacity - player_mass[0] for i in available):
                raise AssertionError("solver passed while a fitting item remained")
        else:
            item = action - 1
            if item not in available:
                raise AssertionError("solver selected an unavailable or invalid item")
            if player_mass[0] + masses[item] > capacity:
                raise AssertionError("solver exceeded its capacity")
            available.remove(item)
            player_mass[0] += masses[item]
            player_value[0] += values[item]

        bot_actions = []
        for j in range(1, k):
            legal = [
                i for i in available
                if player_mass[j] + masses[i] <= capacity
            ]
            if not legal:
                bot_actions.append(0)
                continue
            item = min(legal, key=lambda i: order.index(i))
            available.remove(item)
            player_mass[j] += masses[item]
            player_value[j] += values[item]
            bot_actions.append(item + 1)

        process.stdin.write(" ".join(map(str, bot_actions)) + "\n")
        process.stdin.flush()

    process.stdin.close()
    return_code = process.wait(timeout=5)
    stderr = process.stderr.read() if process.stderr else ""
    extra = process.stdout.read()
    if return_code != 0:
        raise AssertionError(f"solver exited with code {return_code}; stderr={stderr}")
    if extra.strip():
        raise AssertionError(f"unexpected extra stdout after T actions: {extra!r}")

    return {
        "actions": outputs,
        "hidden_policy": hidden_policy,
        "own_mass": player_mass[0],
        "own_value": player_value[0],
        "max_other_mass": max(player_mass[1:]),
        "score": player_value[0] if player_mass[0] < max(player_mass[1:]) else 0,
        "stderr": stderr,
    }


if __name__ == "__main__":
    print(run_protocol())
