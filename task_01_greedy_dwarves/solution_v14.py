import sys
import math


class GreedyDwarvesAgent:
    """V14: V12 with stronger multiplicative evidence updates (2.0 / 0.5).

    All selection logic, opponent hypotheses, simulations, continuation
    policies, and interaction protocol are unchanged from V12.
    """
    """Bit-mask acceleration of the original five-policy greedy agent
    with four own continuation policies.

    Opponent model, weight updates, passes, candidate order and objective
    are retained from V11. V12 adds own continuation policy 0 (max value).
    """

    def __init__(self, K, N, T, W, m, v):
        self.K, self.N, self.T, self.W = K, N, T, W
        self.m, self.v = m, v
        self.round_no = 0
        self.masses = [0] * K
        self.values = [0] * K

        self.available = set(range(N))

        strategies = (
            lambda i: v[i],
            lambda i: v[i] / m[i],
            lambda i: -m[i],
            lambda i: v[i] / math.sqrt(m[i]),
            lambda i: v[i] - 2 * m[i],
        )
        self.orders = [
            sorted(range(N), key=strategy, reverse=True)
            for strategy in strategies
        ]
        self.P = len(self.orders)

        self.item_bits = [[0] * N for _ in range(self.P)]
        self.capacity_masks = [[0] * (W + 1) for _ in range(self.P)]

        for p, order in enumerate(self.orders):
            for rank, item in enumerate(order):
                bit = 1 << rank
                self.item_bits[p][item] = bit
                if m[item] <= W:
                    self.capacity_masks[p][m[item]] |= bit

            for cap in range(1, W + 1):
                self.capacity_masks[p][cap] |= self.capacity_masks[p][cap - 1]

        full = (1 << N) - 1
        self.remaining_masks = [full] * self.P
        self.weights = [1.0] * self.P

        # V12: add policy 0 (greedy maximum value) to V11's three policies.
        self.own_policies = (0, 1, 2, 3)

    def _pick(self, policy, rank_mask, capacity):
        if capacity < 0:
            return -1
        if capacity > self.W:
            capacity = self.W
        legal = rank_mask & self.capacity_masks[policy][capacity]
        if not legal:
            return -1
        rank = (legal & -legal).bit_length() - 1
        return self.orders[policy][rank]

    def _simulate(self, first_item, round_num, bot_policy, own_policy):
        bot_mask = self.remaining_masks[bot_policy]
        own_mask = self.remaining_masks[own_policy]

        masses = self.masses.copy()
        own_value = self.values[0] + self.v[first_item]
        masses[0] += self.m[first_item]

        bot_mask &= ~self.item_bits[bot_policy][first_item]
        own_mask &= ~self.item_bits[own_policy][first_item]

        # Remaining opponents in the current round act after us.
        for j in range(1, self.K):
            item = self._pick(bot_policy, bot_mask, self.W - masses[j])
            if item == -1:
                continue

            masses[j] += self.m[item]
            bot_mask &= ~self.item_bits[bot_policy][item]
            own_mask &= ~self.item_bits[own_policy][item]

        # In later rounds we move first, then all opponents in order.
        for _ in range(round_num + 1, self.T):
            for j in range(self.K):
                if j == 0:
                    policy = own_policy
                    rank_mask = own_mask
                else:
                    policy = bot_policy
                    rank_mask = bot_mask

                item = self._pick(policy, rank_mask, self.W - masses[j])
                if item == -1:
                    continue

                masses[j] += self.m[item]
                if j == 0:
                    own_value += self.v[item]

                bot_mask &= ~self.item_bits[bot_policy][item]
                own_mask &= ~self.item_bits[own_policy][item]

            if not bot_mask:
                break

        return own_value if masses[0] < max(masses[1:]) else 0

    def choose(self):
        free = self.W - self.masses[0]
        candidates = [i for i in self.available if self.m[i] <= free]
        if not candidates:
            return -1

        best_score = -float("inf")
        chosen = candidates[0]

        for item in candidates:
            score = 0.0

            for s in range(self.P):
                results = [
                    self._simulate(item, self.round_no, s, own_policy)
                    for own_policy in self.own_policies
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

    def commit_own_action(self, item):
        if item == -1:
            return

        self.available.remove(item)
        for p in range(self.P):
            self.remaining_masks[p] &= ~self.item_bits[p][item]

        self.masses[0] += self.m[item]
        self.values[0] += self.v[item]

    def observe(self, actions):
        for j, action in enumerate(actions, start=1):
            actual = action - 1

            for s in range(self.P):
                predicted = self._pick(
                    s,
                    self.remaining_masks[s],
                    self.W - self.masses[j],
                )

                if predicted == actual:
                    self.weights[s] *= 2.0
                else:
                    self.weights[s] *= 0.5

                self.weights[s] = max(self.weights[s], 0.000001)

            if actual != -1:
                self.available.remove(actual)
                for p in range(self.P):
                    self.remaining_masks[p] &= ~self.item_bits[p][actual]

                self.masses[j] += self.m[actual]
                self.values[j] += self.v[actual]

        total = sum(self.weights)
        self.weights = [x / total for x in self.weights]
        self.round_no += 1


def solve():
    read = sys.stdin.buffer.readline
    first = read()
    if not first:
        return

    try:
        K, N, T, W = map(int, first.split())
        m = list(map(int, read().split()))
        v = list(map(int, read().split()))
    except (EOFError, ValueError):
        return

    if len(m) != N or len(v) != N:
        return

    agent = GreedyDwarvesAgent(K, N, T, W, m, v)

    for _ in range(T):
        chosen = agent.choose()
        agent.commit_own_action(chosen)
        print(chosen + 1, flush=True)

        line = read()
        if not line:
            return

        try:
            actions = list(map(int, line.split()))
        except ValueError:
            return

        if len(actions) != K - 1:
            return

        agent.observe(actions)


if __name__ == "__main__":
    solve()
