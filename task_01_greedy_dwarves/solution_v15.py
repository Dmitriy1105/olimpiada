import sys
import math


class GreedyDwarvesAgent:
    """V15: two-round beam search with deterministic opponent rollouts.

    V15 keeps V12's five opponent hypotheses and multiplicative posterior
    updates. It replaces the four fixed own-continuation simulations as the
    main decision mechanism with a short beam search over future own actions.
    """

    ROOT_LIMIT = 14
    BEAM_WIDTH = 4
    BRANCH_FACTOR = 6
    SEARCH_DEPTH = 2

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
        self.own_policies = (0, 1, 2, 3)

    def _pick(self, policy, rank_mask, capacity):
        if capacity <= 0:
            return -1
        if capacity > self.W:
            capacity = self.W
        legal = rank_mask & self.capacity_masks[policy][capacity]
        if not legal:
            return -1
        rank = (legal & -legal).bit_length() - 1
        return self.orders[policy][rank]

    def _remove_item(self, masks, item):
        for p in range(self.P):
            masks[p] &= ~self.item_bits[p][item]

    def _play_opponents(self, masks, masses, bot_policy):
        for j in range(1, self.K):
            item = self._pick(
                bot_policy,
                masks[bot_policy],
                self.W - masses[j],
            )
            if item >= 0:
                masses[j] += self.m[item]
                self._remove_item(masks, item)

    def _rollout(self, masks0, masses0, own_value, bot_policy,
                 own_policy, next_round):
        masks = list(masks0)
        masses = list(masses0)
        value = own_value

        for _ in range(next_round, self.T):
            item = self._pick(
                own_policy,
                masks[own_policy],
                self.W - masses[0],
            )
            if item >= 0:
                masses[0] += self.m[item]
                value += self.v[item]
                self._remove_item(masks, item)

            self._play_opponents(masks, masses, bot_policy)

        return value if masses[0] < max(masses[1:]) else 0

    def _leaf_value(self, state, bot_policy, next_round):
        masks, masses, own_value = state
        results = [
            self._rollout(
                masks, masses, own_value, bot_policy, q, next_round
            )
            for q in self.own_policies
        ]
        return 0.7 * max(results) + 0.3 * sum(results) / len(results)

    def _branch_items(self, masks, free):
        # Diverse proposals: first and second feasible item in each ranking.
        found = []
        seen = set()
        for depth in range(2):
            for p in range(self.P):
                legal = masks[p] & self.capacity_masks[p][free]
                if not legal:
                    continue
                for _ in range(depth + 1):
                    if not legal:
                        break
                    bit = legal & -legal
                    rank = bit.bit_length() - 1
                    item = self.orders[p][rank]
                    legal ^= bit
                if item not in seen:
                    seen.add(item)
                    found.append(item)
                    if len(found) >= self.BRANCH_FACTOR:
                        return found
        return found

    def _advance(self, state, action, bot_policy):
        masks0, masses0, value0 = state
        masks = list(masks0)
        masses = list(masses0)
        value = value0

        if action >= 0:
            masses[0] += self.m[action]
            value += self.v[action]
            self._remove_item(masks, action)

        self._play_opponents(masks, masses, bot_policy)
        return tuple(masks), tuple(masses), value

    def _beam_value(self, root_item, bot_policy):
        masks = list(self.remaining_masks)
        masses = self.masses.copy()
        value = self.values[0] + self.v[root_item]

        masses[0] += self.m[root_item]
        self._remove_item(masks, root_item)

        # Opponents still act in the current round after our root action.
        self._play_opponents(masks, masses, bot_policy)
        beam = [(tuple(masks), tuple(masses), value)]

        depth_limit = min(
            self.SEARCH_DEPTH,
            max(0, self.T - self.round_no - 1),
        )

        for depth in range(depth_limit):
            children = []
            next_round = self.round_no + depth + 2

            for state in beam:
                state_masks, state_masses, _ = state
                free = self.W - state_masses[0]
                actions = self._branch_items(state_masks, free) if free > 0 else []
                if not actions:
                    actions = [-1]

                for action in actions:
                    child = self._advance(state, action, bot_policy)

                    # Two complementary rollouts guide beam pruning.
                    r_value = self._rollout(
                        child[0], child[1], child[2],
                        bot_policy, 0, next_round
                    )
                    r_density = self._rollout(
                        child[0], child[1], child[2],
                        bot_policy, 1, next_round
                    )
                    priority = (
                        0.7 * max(r_value, r_density)
                        + 0.3 * (r_value + r_density) / 2
                    )
                    priority += 0.001 * child[2]
                    priority -= 0.0001 * child[1][0]
                    children.append((priority, child))

            if not children:
                break
            children.sort(key=lambda x: x[0], reverse=True)
            beam = [state for _, state in children[:self.BEAM_WIDTH]]

        next_round = self.round_no + depth_limit + 1
        return max(
            self._leaf_value(state, bot_policy, next_round)
            for state in beam
        )

    def _simulate(self, first_item, round_num, bot_policy, own_policy):
        masks = list(self.remaining_masks)
        masses = self.masses.copy()
        value = self.values[0] + self.v[first_item]

        masses[0] += self.m[first_item]
        self._remove_item(masks, first_item)
        self._play_opponents(masks, masses, bot_policy)

        for _ in range(round_num + 1, self.T):
            item = self._pick(
                own_policy,
                masks[own_policy],
                self.W - masses[0],
            )
            if item >= 0:
                masses[0] += self.m[item]
                value += self.v[item]
                self._remove_item(masks, item)
            self._play_opponents(masks, masses, bot_policy)

        return value if masses[0] < max(masses[1:]) else 0

    def _baseline_score(self, item):
        score = 0.0
        for s in range(self.P):
            results = [
                self._simulate(item, self.round_no, s, q)
                for q in self.own_policies
            ]
            expected = 0.7 * max(results) + 0.3 * sum(results) / len(results)
            score += self.weights[s] * expected
        return score / sum(self.weights)

    def choose(self):
        free = self.W - self.masses[0]
        candidates = [i for i in self.available if self.m[i] <= free]
        if not candidates:
            return -1

        # V12 evaluation provides a conservative shortlist, not the final move.
        baseline = []
        for item in candidates:
            score = self._baseline_score(item)
            score += 0.001 * self.v[item]
            score -= 0.0001 * self.m[item]
            baseline.append((score, item))
        baseline.sort(reverse=True)
        roots = [item for _, item in baseline[:self.ROOT_LIMIT]]

        best_item = roots[0]
        best_score = -float("inf")
        weight_sum = sum(self.weights)

        for item in roots:
            score = 0.0
            for s in range(self.P):
                score += self.weights[s] * self._beam_value(item, s)
            score /= weight_sum
            score += 0.001 * self.v[item]
            score -= 0.0001 * self.m[item]

            if score > best_score:
                best_score = score
                best_item = item

        return best_item

    def commit_own_action(self, item):
        if item < 0:
            return
        self.available.remove(item)
        self._remove_item(self.remaining_masks, item)
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
                    self.weights[s] *= 1.25
                else:
                    self.weights[s] *= 0.85
                self.weights[s] = max(self.weights[s], 0.000001)

            if actual >= 0:
                self.available.remove(actual)
                self._remove_item(self.remaining_masks, actual)
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
