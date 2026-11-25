"""Local protocol smoke test for Task 1 V3."""
import random
import subprocess
import sys
from pathlib import Path

from solution_v3 import GreedyDwarvesAgent

def run(seed=77, k=10, n=100, rounds=20, capacity=50):
    rng = random.Random(seed)
    masses = [rng.randint(1, 50) for _ in range(n)]
    values = [rng.randint(1, 100) for _ in range(n)]
    model = GreedyDwarvesAgent(k, n, rounds, capacity, masses, values, cpu_budget=0.01)
    hidden_policy = rng.randrange(model.p_count)
    order = model.orders[hidden_policy]
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name('solution_v3.py'))],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1
    )
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(f'{k} {n} {rounds} {capacity}\\n')
    process.stdin.write(' '.join(map(str, masses)) + '\\n')
    process.stdin.write(' '.join(map(str, values)) + '\\n')
    process.stdin.flush()
    available = set(range(n))
    player_mass = [0] * k
    player_value = [0] * k
    actions = []
    for _ in range(rounds):
        line = process.stdout.readline()
        if not line:
            raise AssertionError('solver exited before producing T actions')
        action = int(line.strip())
        actions.append(action)
        if action == 0:
            if any(masses[i] <= capacity - player_mass[0] for i in available):
                raise AssertionError('voluntary pass while an item fits')
        else:
            item = action - 1
            if item not in available or player_mass[0] + masses[item] > capacity:
                raise AssertionError('illegal own action')
            available.remove(item)
            player_mass[0] += masses[item]
            player_value[0] += values[item]
        bot_actions = []
        for j in range(1, k):
            free = capacity - player_mass[j]
            item = next((x for x in order if x in available and masses[x] <= free), None)
            if item is None:
                bot_actions.append(0)
            else:
                available.remove(item)
                player_mass[j] += masses[item]
                player_value[j] += values[item]
                bot_actions.append(item + 1)
        process.stdin.write(' '.join(map(str, bot_actions)) + '\\n')
        process.stdin.flush()
    process.stdin.close()
    code = process.wait(timeout=5)
    stderr = process.stderr.read() if process.stderr else ''
    extra = process.stdout.read() if process.stdout else ''
    if code != 0:
        raise AssertionError(f'solver exit code {code}; stderr={stderr}')
    if extra.strip():
        raise AssertionError(f'unexpected extra stdout: {extra!r}')
    max_other = max(player_mass[1:])
    return {'hidden_policy': hidden_policy, 'own_mass': player_mass[0],
            'max_other_mass': max_other, 'own_value': player_value[0],
            'score': player_value[0] if player_mass[0] < max_other else 0,
            'actions': actions}

if __name__ == '__main__':
    print(run())