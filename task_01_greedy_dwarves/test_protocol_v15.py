"""Maximum-size interactive protocol test for V15."""
import importlib.util
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOLVER = ROOT / "solution_v15.py"


def run(seed=20261126, k=10, n=100, rounds=20, capacity=50, hidden=2):
    rng = random.Random(seed)
    masses = [rng.randint(1, 50) for _ in range(n)]
    values = [rng.randint(1, 100) for _ in range(n)]

    spec = importlib.util.spec_from_file_location("v15_model", SOLVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    order = module.GreedyDwarvesAgent(
        k, n, rounds, capacity, masses, values
    ).orders[hidden]

    process = subprocess.Popen(
        [sys.executable, str(SOLVER)],
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
    player_masses = [0] * k
    player_values = [0] * k
    outputs = []

    for _ in range(rounds):
        line = process.stdout.readline()
        if not line:
            raise AssertionError("solver exited before producing T actions")
        action = int(line.strip())
        outputs.append(action)

        if action == 0:
            if any(masses[i] <= capacity - player_masses[0] for i in available):
                raise AssertionError("voluntary pass while a fitting item remained")
        else:
            item = action - 1
            if item not in available:
                raise AssertionError("duplicate or invalid own item")
            if player_masses[0] + masses[item] > capacity:
                raise AssertionError("own capacity exceeded")
            available.remove(item)
            player_masses[0] += masses[item]
            player_values[0] += values[item]

        bot_actions = []
        for player in range(1, k):
            item = next(
                (
                    i for i in order
                    if i in available
                    and player_masses[player] + masses[i] <= capacity
                ),
                -1,
            )
            if item < 0:
                bot_actions.append(0)
            else:
                available.remove(item)
                player_masses[player] += masses[item]
                player_values[player] += values[item]
                bot_actions.append(item + 1)

        process.stdin.write(" ".join(map(str, bot_actions)) + "\n")
        process.stdin.flush()

    process.stdin.close()
    return_code = process.wait(timeout=15)
    stderr = process.stderr.read() if process.stderr else ""
    extra_stdout = process.stdout.read()

    if return_code != 0:
        raise AssertionError(f"solver exit={return_code}; stderr={stderr}")
    if extra_stdout.strip():
        raise AssertionError(f"unexpected extra stdout: {extra_stdout!r}")

    return {
        "turns": len(outputs),
        "return_code": return_code,
        "extra_stdout": extra_stdout,
        "own_mass": player_masses[0],
        "max_other_mass": max(player_masses[1:]),
        "reward": (
            player_values[0]
            if player_masses[0] < max(player_masses[1:])
            else 0
        ),
    }


if __name__ == "__main__":
    print(run())
