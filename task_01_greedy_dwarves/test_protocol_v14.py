"""Subprocess protocol test for V14 at maximum task parameters."""
import random
import subprocess
import sys
from pathlib import Path

SOLVER = Path(__file__).with_name("solution_v14.py")


def run(seed=20261126, k=10, n=100, rounds=20, capacity=50):
    rng = random.Random(seed)
    masses = [rng.randint(1, 50) for _ in range(n)]
    values = [rng.randint(1, 100) for _ in range(n)]

    spec = __import__("importlib.util").util.spec_from_file_location("v14_model", SOLVER)
    module = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.GreedyDwarvesAgent(k, n, rounds, capacity, masses, values)
    hidden_policy = rng.randrange(len(model.orders))

    proc = subprocess.Popen(
        [sys.executable, str(SOLVER)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(f"{k} {n} {rounds} {capacity}\n")
    proc.stdin.write(" ".join(map(str, masses)) + "\n")
    proc.stdin.write(" ".join(map(str, values)) + "\n")
    proc.stdin.flush()

    available = set(range(n))
    player_mass = [0] * k
    player_value = [0] * k
    order = model.orders[hidden_policy]
    outputs = []

    for _ in range(rounds):
        line = proc.stdout.readline()
        if not line:
            raise AssertionError("solver exited before producing T actions")
        action = int(line.strip())
        outputs.append(action)

        if action == 0:
            if any(masses[i] <= capacity - player_mass[0] for i in available):
                raise AssertionError("voluntary pass while an item fits")
        else:
            item = action - 1
            if item not in available:
                raise AssertionError("duplicate or invalid own item")
            if player_mass[0] + masses[item] > capacity:
                raise AssertionError("own capacity exceeded")
            available.remove(item)
            player_mass[0] += masses[item]
            player_value[0] += values[item]

        bot_actions = []
        for player in range(1, k):
            legal = [
                i for i in order
                if i in available and player_mass[player] + masses[i] <= capacity
            ]
            if not legal:
                bot_actions.append(0)
                continue
            item = legal[0]
            available.remove(item)
            player_mass[player] += masses[item]
            player_value[player] += values[item]
            bot_actions.append(item + 1)

        proc.stdin.write(" ".join(map(str, bot_actions)) + "\n")
        proc.stdin.flush()

    proc.stdin.close()
    return_code = proc.wait(timeout=30)
    stderr = proc.stderr.read() if proc.stderr else ""
    extra = proc.stdout.read()
    if return_code != 0:
        raise AssertionError(f"solver exit={return_code}; stderr={stderr}")
    if extra.strip():
        raise AssertionError(f"unexpected extra stdout: {extra!r}")

    return {
        "turns": len(outputs),
        "hidden_policy": hidden_policy,
        "own_mass": player_mass[0],
        "max_other_mass": max(player_mass[1:]),
        "score": player_value[0] if player_mass[0] < max(player_mass[1:]) else 0,
        "return_code": return_code,
        "stderr_empty": not stderr.strip(),
    }


if __name__ == "__main__":
    print(run())
