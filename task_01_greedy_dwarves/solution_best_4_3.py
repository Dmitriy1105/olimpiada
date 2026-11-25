import sys
import math


def solve():
    try:
        K, N, T, W = map(int, input().split())
        m = list(map(int, input().split()))
        v = list(map(int, input().split()))
    except (EOFError, ValueError):
        return

    available = set(range(N))
    masses = [0] * K
    values = [0] * K

    strategies = [
        lambda i: v[i],
        lambda i: v[i] / m[i],
        lambda i: -m[i],
        lambda i: v[i] / math.sqrt(m[i]),
        lambda i: v[i] - 2 * m[i],
    ]

    orders = [
        sorted(range(N), key=strategy, reverse=True)
        for strategy in strategies
    ]

    weights = [1.0] * len(strategies)

    def select_item(items, capacity, order):
        for i in order:
            if i in items and m[i] <= capacity:
                return i
        return -1

    def simulate(first_item, round_num, bot_order, my_order):
        remaining = available.copy()
        M = masses.copy()
        V = values.copy()

        remaining.remove(first_item)
        M[0] += m[first_item]
        V[0] += v[first_item]

        for r in range(round_num, T):
            start = 1 if r == round_num else 0

            for j in range(start, K):
                order = my_order if j == 0 else bot_order
                item = select_item(remaining, W - M[j], order)

                if item == -1:
                    continue

                remaining.remove(item)
                M[j] += m[item]
                V[j] += v[item]

            if not remaining:
                break

        max_other_mass = max(M[1:])
        if M[0] < max_other_mass:
            return V[0]
        return 0

    for r in range(T):
        candidates = [
            i for i in available
            if masses[0] + m[i] <= W
        ]

        if not candidates:
            chosen = -1
        else:
            best_score = -float("inf")
            chosen = candidates[0]

            for item in candidates:
                score = 0.0

                for s, bot_order in enumerate(orders):
                    results = [
                        simulate(item, r, bot_order, orders[1]),
                        simulate(item, r, bot_order, orders[2]),
                        simulate(item, r, bot_order, orders[3]),
                    ]

                    expected = (
                        0.7 * max(results)
                        + 0.3 * sum(results) / len(results)
                    )
                    score += weights[s] * expected

                score /= sum(weights)
                score += 0.001 * v[item]
                score -= 0.0001 * m[item]

                if score > best_score:
                    best_score = score
                    chosen = item

        print(chosen + 1, flush=True)

        if chosen != -1:
            available.remove(chosen)
            masses[0] += m[chosen]
            values[0] += v[chosen]

        line = sys.stdin.readline()
        if not line:
            return

        actions = list(map(int, line.split()))
        if len(actions) != K - 1:
            return

        for j, action in enumerate(actions, start=1):
            actual = action - 1

            for s, order in enumerate(orders):
                predicted = select_item(
                    available,
                    W - masses[j],
                    order
                )

                if predicted == actual:
                    weights[s] *= 1.25
                else:
                    weights[s] *= 0.85

                weights[s] = max(weights[s], 0.000001)

            if actual != -1:
                available.remove(actual)
                masses[j] += m[actual]
                values[j] += v[actual]

        total = sum(weights)
        weights = [x / total for x in weights]


if __name__ == "__main__":
    solve()
