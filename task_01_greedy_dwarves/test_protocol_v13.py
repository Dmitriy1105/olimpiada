"""Maximum-size subprocess protocol test for solution_v13.py."""
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOLVER = ROOT / "solution_v13.py"


def run(seed=20261126, k=10, n=100, rounds=20, capacity=50):
    rng = random.Random(seed)
    masses = [rng.randint(1, 50) for _ in range(n)]
    values = [rng.randint(1, 100) for _ in range(n)]

    spec = __import__("importlib.util").util.spec_from_file_location(
        "v13_protocol_model", SOLVER
    )
    module = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.GreedyDwarvesAgent(k, n, rounds, capacity, masses, values)
    hidden_policy = rng.randrange(len(model.orders))

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
    player_mass = [0] * k
    player_value = [0] * k
    order = model.orders[hidden_policy]
    outputs = []

    for _ in range(rounds):
        line = process.stdout.readline()
        if not line:
            raise AssertionError("solver exited before producing T actions")
        action = int(line.strip())
        outputs.append(action)

        if action == 0:
            if any(masses[i] <= capacity - player_mass[0] for i in available):
                raise AssertionError("solver passed while a fitting item remained")
        else:
            item = action - 1
            if item not in available:
                raise AssertionError("solver selected an unavailable item")
            if player_mass[0] + masses[item] > capacity:
                raise AssertionError("solver exceeded capacity")
            available.remove(item)
            player_mass[0] += masses[item]
            player_value[0] += values[item]

        bot_actions = []
        for player in range(1, k):
            item = -1
            for candidate in order:
                if (
                    candidate in available
                    and player_mass[player] + masses[candidate] <= capacity
                ):
                    item = candidate
                    break
            if item < 0:
                bot_actions.append(0)
            else:
                available.remove(item)
                player_mass[player] += masses[item]
                player_value[player] += values[item]
                bot_actions.append(item + 1)

        process.stdin.write(" ".join(map(str, bot_actions)) + "\n")
        process.stdin.flush()

    process.stdin.close()
    return_code = process.wait(timeout=30)
    stderr = process.stderr.read() if process.stderr else ""
    extra_stdout = process.stdout.read()

    if return_code != 0:
        raise AssertionError(f"nonzero exit code {return_code}; stderr={stderr}")
    if extra_stdout.strip():
        raise AssertionError(f"unexpected extra stdout: {extra_stdout!r}")

    return {
        "turns": len(outputs),
        "hidden_policy": hidden_policy,
        "own_mass": player_mass[0],
        "max_other_mass": max(player_mass[1:]),
        "score": (
            player_value[0]
            if player_mass[0] < max(player_mass[1:])
            else 0
        ),
        "return_code": return_code,
        "stderr_empty": not stderr.strip(),
    }


if __name__ == "__main__":
    print(run())
