import sys
import math


class GreedyDwarvesAgent:
    def __init__(self, K, N, T, W, m, v):
        self.K, self.N, self.T, self.W = K, N, T, W
        self.m, self.v = m, v
        self.round_no = 0
        self.masses = [0] * K
        self.values = [0] * K
        self.available_mask = (1 << N) - 1

        # Original five hypotheses from the best 4.3-point solution,
        # plus three targeted alternatives.
        key_functions = [
            lambda i: v[i],
            lambda i: v[i] / m[i],
            lambda i: -m[i],
            lambda i: v[i] / math.sqrt(m[i]),
            lambda i: v[i] - 2.0 * m[i],
            lambda i: v[i] - 1.5 * m[i],
            lambda i: v[i] - 3.0 * m[i],
            lambda i: v[i] / (m[i] + 5.0),
        ]
        self.orders = [
            sorted(range(N), key=key, reverse=True)
            for key in key_functions
        ]
        self.P = len(self.orders)

        # A bit in a policy mask denotes a rank, not an item id.
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

        # All opponents in one test share one hidden strategy.
        self.posterior = [1.0 / self.P] * self.P

        # Keep the three continuation policies from the best submission.
        self.own_policies = (1, 2, 3)

        # Soft-rank likelihood: rank 0 is strongest evidence, but
        # near-optimal actions are not treated as impossible.
        self.beta = 0.7
        self.lapse = 0.02
        self.exp_rank = [math.exp(-self.beta * r) for r in range(N + 1)]

    def _pick(self, policy, rank_mask, free_capacity):
        if free_capacity <= 0:
            return -1
        free_capacity = min(free_capacity, self.W)
        legal = rank_mask & self.capacity_masks[policy][free_capacity]
        if not legal:
            return -1
        rank = (legal & -legal).bit_length() - 1
        return self.orders[policy][rank]

    def _simulate(self, first_item, bot_policy, own_policy):
        bot_mask = self.remaining_masks[bot_policy]
        own_mask = self.remaining_masks[own_policy]

        masses = self.masses.copy()
        own_value = self.values[0] + self.v[first_item]
        masses[0] += self.m[first_item]

        # Filling the backpack to W guarantees a loss: ties do not win.
        if masses[0] >= self.W:
            return 0

        bot_mask &= ~self.item_bits[bot_policy][first_item]
        own_mask &= ~self.item_bits[own_policy][first_item]

        # Remaining opponents in the current round act after us.
        for j in range(1, self.K):
            item = self._pick(bot_policy, bot_mask, self.W - masses[j])
            if item >= 0:
                masses[j] += self.m[item]
                bot_mask &= ~self.item_bits[bot_policy][item]
                own_mask &= ~self.item_bits[own_policy][item]

        # In later rounds we move first, then all opponents in order.
        for _ in range(self.round_no + 1, self.T):
            changed = False

            item = self._pick(own_policy, own_mask, self.W - masses[0])
            if item >= 0:
                masses[0] += self.m[item]
                own_value += self.v[item]
                bot_mask &= ~self.item_bits[bot_policy][item]
                own_mask &= ~self.item_bits[own_policy][item]
                changed = True
                if masses[0] >= self.W:
                    return 0

            for j in range(1, self.K):
                item = self._pick(bot_policy, bot_mask, self.W - masses[j])
                if item >= 0:
                    masses[j] += self.m[item]
                    bot_mask &= ~self.item_bits[bot_policy][item]
                    own_mask &= ~self.item_bits[own_policy][item]
                    changed = True

            # If nobody can take an item now, no later round can change that.
            if not changed:
                break

        return own_value if masses[0] < max(masses[1:]) else 0

    def choose(self):
        free = self.W - self.masses[0]
        candidates = [
            i for i in range(self.N)
            if ((self.available_mask >> i) & 1) and self.m[i] <= free
        ]
        if not candidates:
            return -1

        best_item = candidates[0]
        best_score = -float("inf")

        for item in candidates:
            score = 0.0
            for s in range(self.P):
                results = [
                    self._simulate(item, s, own_policy)
                    for own_policy in self.own_policies
                ]
                expected = 0.7 * max(results) + 0.3 * sum(results) / len(results)
                score += self.posterior[s] * expected

            # Preserve the tie-break terms from the best submission.
            score += 0.001 * self.v[item]
            score -= 0.0001 * self.m[item]

            if score > best_score:
                best_score = score
                best_item = item

        return best_item

    def commit_own_action(self, item):
        if item < 0:
            return
        self.available_mask &= ~(1 << item)
        for p in range(self.P):
            self.remaining_masks[p] &= ~self.item_bits[p][item]
        self.masses[0] += self.m[item]
        self.values[0] += self.v[item]

    def observe(self, actions):
        for j, action in enumerate(actions, start=1):
            if action == 0:
                continue

            item = action - 1
            if not (0 <= item < self.N):
                raise ValueError("invalid opponent item")
            if not ((self.available_mask >> item) & 1):
                raise ValueError("opponent selected unavailable item")

            free = self.W - self.masses[j]
            if self.m[item] > free:
                raise ValueError("opponent exceeded capacity")

            # All policies have the same number L of feasible items.
            legal0 = self.remaining_masks[0] & self.capacity_masks[0][free]
            L = legal0.bit_count()
            if L > 0:
                z = (1.0 - self.exp_rank[L]) / (1.0 - math.exp(-self.beta))
                uniform = self.lapse / L

                for p in range(self.P):
                    legal = self.remaining_masks[p] & self.capacity_masks[p][free]
                    bit = self.item_bits[p][item]
                    if not (legal & bit):
                        likelihood = uniform
                    else:
                        rank = (legal & (bit - 1)).bit_count()
                        likelihood = (
                            (1.0 - self.lapse) * self.exp_rank[rank] / z
                            + uniform
                        )
                    self.posterior[p] *= likelihood

                total = sum(self.posterior)
                if total > 0.0:
                    self.posterior = [x / total for x in self.posterior]
                else:
                    self.posterior = [1.0 / self.P] * self.P

            self.available_mask &= ~(1 << item)
            for p in range(self.P):
                self.remaining_masks[p] &= ~self.item_bits[p][item]
            self.masses[j] += self.m[item]
            self.values[j] += self.v[item]

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

        try:
            agent.observe(actions)
        except ValueError:
            return


if __name__ == "__main__":
    solve()
