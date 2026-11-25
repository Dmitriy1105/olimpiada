import random
import math

from solution_v11 import GreedyDwarvesAgent


class ReferenceAgent:
    """Literal set-based implementation of the user's original algorithm."""

    def __init__(self, K, N, T, W, m, v):
        self.K, self.N, self.T, self.W = K, N, T, W
        self.m, self.v = m, v
        self.round_no = 0
        self.available = set(range(N))
        self.masses = [0] * K
        self.values = [0] * K
        strategies = [
            lambda i: v[i],
            lambda i: v[i] / m[i],
            lambda i: -m[i],
            lambda i: v[i] / math.sqrt(m[i]),
            lambda i: v[i] - 2 * m[i],
        ]
        self.orders = [
            sorted(range(N), key=f, reverse=True) for f in strategies
        ]
        self.weights = [1.0] * len(self.orders)

    def select_item(self, items, capacity, order):
        for i in order:
            if i in items and self.m[i] <= capacity:
                return i
        return -1

    def simulate(self, first_item, round_num, bot_order, my_order):
        remaining = self.available.copy()
        M = self.masses.copy()
        V = self.values.copy()
        remaining.remove(first_item)
        M[0] += self.m[first_item]
        V[0] += self.v[first_item]

        for r in range(round_num, self.T):
            start = 1 if r == round_num else 0
            for j in range(start, self.K):
                order = my_order if j == 0 else bot_order
                item = self.select_item(remaining, self.W - M[j], order)
                if item == -1:
                    continue
                remaining.remove(item)
                M[j] += self.m[item]
                V[j] += self.v[item]
            if not remaining:
                break

        return V[0] if M[0] < max(M[1:]) else 0

    def choose(self):
        candidates = [
            i for i in self.available
            if self.masses[0] + self.m[i] <= self.W
        ]
        if not candidates:
            return -1

        best_score = -float("inf")
        chosen = candidates[0]
        for item in candidates:
            score = 0.0
            for s, bot_order in enumerate(self.orders):
                results = [
                    self.simulate(item, self.round_no, bot_order, self.orders[1]),
                    self.simulate(item, self.round_no, bot_order, self.orders[2]),
                    self.simulate(item, self.round_no, bot_order, self.orders[3]),
                ]
                expected = (
                    0.7 * max(results)
                    + 0.3 * sum(results) / len(results)
                )
                score += self.weights[s] * expected

            score /= sum(self.weights)
            score += 0.001 * self.v[item]
            score -= 0.0001 * self.m[item]
            if score > best_score:
                best_score = score
                chosen = item
        return chosen

    def commit(self, item):
        if item != -1:
            self.available.remove(item)
            self.masses[0] += self.m[item]
            self.values[0] += self.v[item]

    def observe(self, actions):
        for j, action in enumerate(actions, start=1):
            actual = action - 1
            for s, order in enumerate(self.orders):
                predicted = self.select_item(
                    self.available, self.W - self.masses[j], order
                )
                if predicted == actual:
                    self.weights[s] *= 1.25
                else:
                    self.weights[s] *= 0.85
                self.weights[s] = max(self.weights[s], 0.000001)

            if actual != -1:
                self.available.remove(actual)
                self.masses[j] += self.m[actual]
                self.values[j] += self.v[actual]

        total = sum(self.weights)
        self.weights = [x / total for x in self.weights]
        self.round_no += 1


def check_simulation_equivalence(seed=1126, states=300):
    rng = random.Random(seed)
    checked = 0
    for _ in range(states):
        K = rng.randint(2, 10)
        N = rng.randint(10, 70)
        T = rng.randint(5, 20)
        W = rng.randint(10, 50)
        m = [rng.randint(1, 50) for _ in range(N)]
        v = [rng.randint(1, 100) for _ in range(N)]
        ref = ReferenceAgent(K, N, T, W, m, v)
        fast = GreedyDwarvesAgent(K, N, T, W, m, v)

        removed = rng.sample(range(N), rng.randint(0, min(N - 1, 12)))
        for item in removed:
            ref.available.remove(item)
            fast.available.remove(item)
            for p in range(fast.P):
                fast.remaining_masks[p] &= ~fast.item_bits[p][item]

        ref.masses = [rng.randint(0, W) for _ in range(K)]
        fast.masses = ref.masses.copy()
        ref.values = [rng.randint(0, 500) for _ in range(K)]
        fast.values = ref.values.copy()
        ref.round_no = fast.round_no = rng.randrange(T)

        candidates = [
            i for i in ref.available
            if ref.masses[0] + m[i] <= W
        ]
        for first in candidates:
            for bp in range(5):
                for op in (1, 2, 3):
                    expected = ref.simulate(
                        first, ref.round_no, ref.orders[bp], ref.orders[op]
                    )
                    actual = fast._simulate(first, ref.round_no, bp, op)
                    assert expected == actual, (
                        expected, actual, K, N, W, first, bp, op
                    )
                    checked += 1
    return checked


def check_full_game_equivalence(seed=2233, games=12):
    rng = random.Random(seed)
    decisions = 0
    pass_updates = 0
    for _ in range(games):
        K = rng.randint(2, 10)
        N = rng.randint(10, 100)
        T = rng.randint(5, 20)
        W = rng.randint(10, 50)
        m = [rng.randint(1, 50) for _ in range(N)]
        v = [rng.randint(1, 100) for _ in range(N)]
        ref = ReferenceAgent(K, N, T, W, m, v)
        fast = GreedyDwarvesAgent(K, N, T, W, m, v)
        hidden = rng.randrange(5)

        for _round in range(T):
            expected = ref.choose()
            actual = fast.choose()
            assert expected == actual, ("choice mismatch", _round, expected, actual)
            decisions += 1
            ref.commit(expected)
            fast.commit_own_action(actual)

            actions = []
            simulated_available = ref.available.copy()
            simulated_masses = ref.masses.copy()
            for j in range(1, K):
                if rng.random() < 0.12:
                    actions.append(0)
                    pass_updates += 1
                else:
                    item = ref.select_item(
                        simulated_available,
                        W - simulated_masses[j],
                        ref.orders[hidden],
                    )
                    actions.append(item + 1)
                    if item >= 0:
                        simulated_available.remove(item)
                        simulated_masses[j] += m[item]

            ref.observe(actions)
            fast.observe(actions)
            assert ref.available == fast.available
            assert ref.masses == fast.masses
            assert ref.values == fast.values
            assert all(
                abs(x - y) < 1e-12
                for x, y in zip(ref.weights, fast.weights)
            ), (ref.weights, fast.weights)

    return decisions, pass_updates


if __name__ == "__main__":
    simulation_checks = check_simulation_equivalence()
    decisions, pass_updates = check_full_game_equivalence()
    print(f"simulation equivalence: PASS ({simulation_checks} comparisons)")
    print(
        f"full games: PASS ({decisions} decisions, "
        f"{pass_updates} explicit pass observations)"
    )
