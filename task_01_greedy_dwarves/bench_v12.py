import importlib.util
import random
import statistics


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GreedyDwarvesAgent


A11 = load("v11", "solution_v11.py")
A12 = load("v12", "solution_v12.py")


def play(seed, Agent):
    rng = random.Random(seed)
    K = rng.randint(3, 8)
    N = rng.randint(35, 75)
    T = rng.randint(8, 16)
    W = rng.randint(15, 45)

    m = [rng.randint(1, 50) for _ in range(N)]
    v = [rng.randint(1, 100) for _ in range(N)]
    agent = Agent(K, N, T, W, m, v)

    hidden = rng.randrange(5)

    for _ in range(T):
        chosen = agent.choose()
        agent.commit_own_action(chosen)

        temp_available = agent.available.copy()
        temp_masses = agent.masses.copy()
        actions = []

        for j in range(1, K):
            item = -1
            for i in agent.orders[hidden]:
                if i in temp_available and m[i] <= W - temp_masses[j]:
                    item = i
                    break

            actions.append(item + 1)
            if item >= 0:
                temp_available.remove(item)
                temp_masses[j] += m[item]

        agent.observe(actions)

    return (
        agent.values[0]
        if agent.masses[0] < max(agent.masses[1:])
        else 0
    )


if __name__ == "__main__":
    seeds = range(100)
    v11 = [play(seed, A11) for seed in seeds]
    v12 = [play(seed, A12) for seed in seeds]

    print("V11 mean reward:", statistics.mean(v11))
    print("V12 mean reward:", statistics.mean(v12))
    print("V11 positive games:", sum(x > 0 for x in v11), "/ 100")
    print("V12 positive games:", sum(x > 0 for x in v12), "/ 100")
    print("V12 wins:", sum(y > x for x, y in zip(v11, v12)))
    print("V12 losses:", sum(y < x for x, y in zip(v11, v12)))
    print("Ties:", sum(y == x for x, y in zip(v11, v12)))
