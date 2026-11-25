"""Interactive solver for 'Greedy Dwarves' — Python 3.11+.

All opponents in one test share one hidden strategy. We therefore maintain
one common posterior over candidate strategies and simulate every opponent
with the same strategy hypothesis in each rollout.
"""
from __future__ import annotations

import math
import sys
import time


class GreedyDwarvesAgent:
    def __init__(
        self,
        k: int,
        n: int,
        rounds: int,
        capacity: int,
        masses: list[int],
        values: list[int],
        cpu_budget: float = 7.5,
    ) -> None:
        self.k = k
        self.n = n
        self.rounds = rounds
        self.capacity = capacity
        self.m = masses
        self.v = values
        self.round_no = 0
        self.cpu_deadline = time.process_time() + cpu_budget

        # Candidate hypotheses for the single hidden strategy used by all bots.
        keys = [
            lambda i: (-values[i], masses[i], i),                         # maximum value
            lambda i: (-values[i] / masses[i], -values[i], i),            # value / mass
            lambda i: (-values[i] / math.sqrt(masses[i]), -values[i], i), # value / sqrt(mass)
            lambda i: (-values[i] / (masses[i] ** 0.7), -values[i], i),   # power density
            lambda i: (-(values[i] - 1.5 * masses[i]), masses[i], i),     # linear penalty
            lambda i: (-(values[i] - 3.0 * masses[i]), masses[i], i),
            lambda i: (masses[i], -values[i], i),                         # minimum mass
            lambda i: (-masses[i], -values[i], i),                        # maximum mass
            lambda i: (-values[i] / (masses[i] + 5), -values[i], i),
            lambda i: (-values[i] / (masses[i] + 10), -values[i], i),
        ]
        self.orders = [sorted(range(n), key=key) for key in keys]
        self.p_count = len(self.orders)

        # In each policy's order, bit rank r represents orders[p][r].
        self.item_bits = [[0] * n for _ in range(self.p_count)]
        self.capacity_masks = [[0] * (capacity + 1) for _ in range(self.p_count)]
        for p, order in enumerate(self.orders):
            for rank, item in enumerate(order):
                bit = 1 << rank
                self.item_bits[p][item] = bit
                if masses[item] <= capacity:
                    self.capacity_masks[p][masses[item]] |= bit
            for c in range(1, capacity + 1):
                self.capacity_masks[p][c] |= self.capacity_masks[p][c - 1]

        full = (1 << n) - 1
        self.remaining_masks = [full] * self.p_count
        self.available_mask = full
        self.player_mass = [0] * k
        self.player_value = [0] * k

        # One shared posterior: the statement says all opponents use one strategy.
        self.posterior = [1.0 / self.p_count] * self.p_count

        # Candidate policies for our own continuation in rollouts.
        self.own_policies = tuple(range(self.p_count))

    def _pick(self, policy: int, rank_mask: int, free_capacity: int) -> int:
        if free_capacity <= 0:
            return -1
        if free_capacity > self.capacity:
            free_capacity = self.capacity
        legal = rank_mask & self.capacity_masks[policy][free_capacity]
        if not legal:
            return -1
        rank = (legal & -legal).bit_length() - 1
        return self.orders[policy][rank]

    def _update_posterior(self, actual_item: int, free_capacity: int) -> None:
        if free_capacity <= 0:
            return
        free_capacity = min(free_capacity, self.capacity)

        likelihoods = []
        for p in range(self.p_count):
            legal = (
                self.remaining_masks[p]
                & self.capacity_masks[p][free_capacity]
            )
            bit = self.item_bits[p][actual_item]
            if not (legal & bit):
                likelihoods.append(1e-10)
                continue

            # Rank 0 is the exact greedy choice; lower-ranked choices are possible
            # under a noisy/approximate implementation of the same broad heuristic.
            rank = (legal & (bit - 1)).bit_count()
            likelihoods.append(math.exp(-0.8 * rank) + 0.001)

        weighted = [
            self.posterior[p] * likelihoods[p]
            for p in range(self.p_count)
        ]
        total = sum(weighted)
        if total <= 1e-300:
            self.posterior = [1.0 / self.p_count] * self.p_count
            return

        # A small floor prevents one observation from eliminating every alternative.
        self.posterior = [
            0.99 * weighted[p] / total + 0.01 / self.p_count
            for p in range(self.p_count)
        ]

    def _simulate(self, first_item: int, bot_policy: int, own_policy: int) -> int:
        # Only the two relevant policy masks are copied and updated in a rollout.
        bot_mask = self.remaining_masks[bot_policy]
        own_mask = self.remaining_masks[own_policy]
        bot_bits = self.item_bits[bot_policy]
        own_bits = self.item_bits[own_policy]
        bot_capacity = self.capacity_masks[bot_policy]
        own_capacity = self.capacity_masks[own_policy]

        masses = self.player_mass.copy()
        own_value = self.player_value[0] + self.v[first_item]
        masses[0] += self.m[first_item]

        # If our bag reaches W, nobody can be heavier, so strict survival is impossible.
        if masses[0] >= self.capacity:
            return 0

        bot_mask &= ~bot_bits[first_item]
        own_mask &= ~own_bits[first_item]

        # Remaining bots act in this round, after our already-completed move.
        for j in range(1, self.k):
            free = self.capacity - masses[j]
            if free <= 0:
                continue
            legal = bot_mask & bot_capacity[free]
            if legal:
                rank = (legal & -legal).bit_length() - 1
                item = self.orders[bot_policy][rank]
                masses[j] += self.m[item]
                bot_mask &= ~bot_bits[item]
                own_mask &= ~own_bits[item]

        # Future rounds: our move first, then all opponents in fixed order.
        for _ in range(self.round_no + 1, self.rounds):
            changed = False

            free = self.capacity - masses[0]
            if free > 0:
                legal = own_mask & own_capacity[free]
                if legal:
                    rank = (legal & -legal).bit_length() - 1
                    item = self.orders[own_policy][rank]
                    masses[0] += self.m[item]
                    own_value += self.v[item]
                    bot_mask &= ~bot_bits[item]
                    own_mask &= ~own_bits[item]
                    changed = True
                    if masses[0] >= self.capacity:
                        return 0

            for j in range(1, self.k):
                free = self.capacity - masses[j]
                if free <= 0:
                    continue
                legal = bot_mask & bot_capacity[free]
                if legal:
                    rank = (legal & -legal).bit_length() - 1
                    item = self.orders[bot_policy][rank]
                    masses[j] += self.m[item]
                    bot_mask &= ~bot_bits[item]
                    own_mask &= ~own_bits[item]
                    changed = True

            if not changed:
                break

        return own_value if masses[0] < max(masses[1:]) else 0

    def choose(self) -> int:
        candidates = [
            i for i in range(self.n)
            if ((self.available_mask >> i) & 1)
            and self.player_mass[0] + self.m[i] <= self.capacity
        ]
        if not candidates:
            return -1

        # Taking an item that fills the bag to exactly W guarantees zero points.
        non_filling = [
            i for i in candidates
            if self.player_mass[0] + self.m[i] < self.capacity
        ]
        if non_filling:
            candidates = non_filling

        # This order is only a safe fallback if the CPU budget is exhausted.
        candidates.sort(
            key=lambda i: (
                self.v[i] / math.sqrt(self.m[i]),
                self.v[i],
                -self.m[i],
            ),
            reverse=True,
        )

        now = time.process_time()
        remaining = max(0.0, self.cpu_deadline - now)
        turns_left = max(1, self.rounds - self.round_no)
        turn_deadline = now + 0.90 * remaining / turns_left

        best_item = candidates[0]
        best_score = -1.0

        for item in candidates:
            if time.process_time() >= turn_deadline:
                break

            # Choose the best continuation policy by posterior expected final score.
            best_continuation = 0.0
            for own_policy in self.own_policies:
                expected = 0.0
                for bot_policy, probability in enumerate(self.posterior):
                    if probability > 1e-7:
                        expected += probability * self._simulate(
                            item, bot_policy, own_policy
                        )
                if expected > best_continuation:
                    best_continuation = expected

            score = best_continuation + 1e-8 * self.v[item]
            if score > best_score:
                best_score = score
                best_item = item

        return best_item

    def commit_own_action(self, item: int) -> None:
        if item < 0:
            # Passing is legal only if no remaining item fits.
            if any(
                ((self.available_mask >> i) & 1)
                and self.m[i] <= self.capacity - self.player_mass[0]
                for i in range(self.n)
            ):
                raise ValueError("voluntary pass while a fitting item remains")
            return

        if not (0 <= item < self.n):
            raise ValueError("invalid own item index")
        if not ((self.available_mask >> item) & 1):
            raise ValueError("own item is no longer available")
        if self.player_mass[0] + self.m[item] > self.capacity:
            raise ValueError("own item exceeds backpack capacity")

        self.available_mask &= ~(1 << item)
        for p in range(self.p_count):
            self.remaining_masks[p] &= ~self.item_bits[p][item]
        self.player_mass[0] += self.m[item]
        self.player_value[0] += self.v[item]

    def observe(self, actions: list[int]) -> None:
        if len(actions) != self.k - 1:
            raise ValueError("wrong number of opponent actions")

        for j, action in enumerate(actions, start=1):
            if action == 0:
                continue

            item = action - 1
            if not (0 <= item < self.n):
                raise ValueError("invalid opponent item index")
            if not ((self.available_mask >> item) & 1):
                raise ValueError("opponent selected an unavailable item")

            self._update_posterior(
                item,
                self.capacity - self.player_mass[j],
            )

            self.available_mask &= ~(1 << item)
            for p in range(self.p_count):
                self.remaining_masks[p] &= ~self.item_bits[p][item]
            self.player_mass[j] += self.m[item]
            self.player_value[j] += self.v[item]

        self.round_no += 1


def solve() -> None:
    read = sys.stdin.buffer.readline
    first = read()
    if not first:
        return

    try:
        k, n, rounds, capacity = map(int, first.split())
        masses = list(map(int, read().split()))
        values = list(map(int, read().split()))
    except (ValueError, EOFError):
        return

    if len(masses) != n or len(values) != n:
        return

    agent = GreedyDwarvesAgent(k, n, rounds, capacity, masses, values)

    for _ in range(rounds):
        item = agent.choose()
        agent.commit_own_action(item)
        print(item + 1, flush=True)

        line = read()
        if not line:
            return
        try:
            actions = list(map(int, line.split()))
        except ValueError:
            return

        try:
            agent.observe(actions)
        except ValueError:
            return


if __name__ == "__main__":
    solve()
