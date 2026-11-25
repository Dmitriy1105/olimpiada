"""Interactive solver for the problem 'Greedy Dwarves'. Compatible with Python 3.11+."""
from __future__ import annotations

import math
import random
import sys
import time
from typing import Iterable


class GreedyDwarvesAgent:
    """Online agent with a separate opponent-strategy posterior for each dwarf."""

    def __init__(
        self,
        k: int,
        n: int,
        rounds: int,
        capacity: int,
        masses: list[int],
        values: list[int],
        scenario_count: int = 12,
        cpu_budget: float = 8.0,
    ) -> None:
        self.k = k
        self.n = n
        self.rounds = rounds
        self.capacity = capacity
        self.m = masses
        self.v = values
        self.scenario_count = scenario_count
        self.cpu_deadline = time.process_time() + cpu_budget
        self.round_no = 0

        # Plausible deterministic greedy rules. Every model is a fixed item order;
        # an item is selected only if it fits the dwarf's remaining capacity.
        self.orders: list[list[int]] = [
            sorted(range(n), key=lambda i: (-values[i], masses[i], i)),
            sorted(range(n), key=lambda i: (-values[i] / masses[i], -values[i], i)),
            sorted(range(n), key=lambda i: (-values[i] / math.sqrt(masses[i]), -values[i], i)),
            sorted(range(n), key=lambda i: (-values[i] / (masses[i] ** 0.7), -values[i], i)),
            sorted(range(n), key=lambda i: (-(values[i] - 1.5 * masses[i]), masses[i], i)),
            sorted(range(n), key=lambda i: (-(values[i] - 3.0 * masses[i]), masses[i], i)),
            sorted(range(n), key=lambda i: (masses[i], -values[i], i)),
            sorted(range(n), key=lambda i: (-values[i], -masses[i], i)),
        ]
        self.p = len(self.orders)

        # For each model, item -> bit at its rank in that model's order.
        self.item_bits: list[list[int]] = [[0] * n for _ in range(self.p)]
        # capacity_masks[p][c] contains all items of mass <= c, encoded by rank.
        self.capacity_masks: list[list[int]] = [
            [0] * (capacity + 1) for _ in range(self.p)
        ]
        for p, order in enumerate(self.orders):
            for rank, item in enumerate(order):
                bit = 1 << rank
                self.item_bits[p][item] = bit
                if masses[item] <= capacity:
                    self.capacity_masks[p][masses[item]] |= bit
            for c in range(1, capacity + 1):
                self.capacity_masks[p][c] |= self.capacity_masks[p][c - 1]

        self.remaining_masks = [(1 << n) - 1 for _ in range(self.p)]
        self.available_mask = (1 << n) - 1
        self.player_mass = [0] * k
        self.player_value = [0] * k

        # Crucial: each opponent has an independent belief distribution.
        self.posterior = [[1.0 / self.p] * self.p for _ in range(k - 1)]

        # We test several plausible continuation policies for ourselves.
        self.own_policies = (0, 1, 2, 4, 6)

    def _remove(self, item: int, masks: list[int]) -> None:
        for p in range(self.p):
            masks[p] &= ~self.item_bits[p][item]

    def _pick(self, policy: int, masks: list[int], free_capacity: int) -> int:
        if free_capacity <= 0:
            return -1
        if free_capacity > self.capacity:
            free_capacity = self.capacity
        legal = masks[policy] & self.capacity_masks[policy][free_capacity]
        if not legal:
            return -1
        rank = (legal & -legal).bit_length() - 1
        return self.orders[policy][rank]

    def _update_posterior(self, opponent_index: int, actual: int, free_capacity: int) -> None:
        if actual < 0 or free_capacity <= 0:
            return
        if free_capacity > self.capacity:
            free_capacity = self.capacity

        old = self.posterior[opponent_index]
        likelihoods = [0.0] * self.p
        for p in range(self.p):
            legal = self.remaining_masks[p] & self.capacity_masks[p][free_capacity]
            bit = self.item_bits[p][actual]
            if not (legal & bit):
                likelihoods[p] = 1e-8
                continue
            rank = (legal & (bit - 1)).bit_count()
            # A model strongly prefers rank 0, but non-top choices remain possible.
            likelihoods[p] = 0.58 * math.exp(-0.48 * rank) + 0.015

        updated = [old[p] * likelihoods[p] for p in range(self.p)]
        total = sum(updated)
        if total <= 1e-300:
            self.posterior[opponent_index] = [1.0 / self.p] * self.p
            return

        # Small uniform floor prevents a single early observation from locking a model.
        self.posterior[opponent_index] = [
            0.985 * updated[p] / total + 0.015 / self.p
            for p in range(self.p)
        ]

    def _make_scenarios(self) -> list[tuple[int, ...]]:
        rng = random.Random(910_247 + self.round_no * 1_000_003)
        scenarios: list[tuple[int, ...]] = []
        # Include the joint MAP assignment as a deterministic anchor.
        scenarios.append(tuple(max(range(self.p), key=row.__getitem__) for row in self.posterior))
        for _ in range(max(0, self.scenario_count - 1)):
            assignment = []
            for row in self.posterior:
                x = rng.random()
                cumulative = 0.0
                chosen = self.p - 1
                for p, probability in enumerate(row):
                    cumulative += probability
                    if x <= cumulative:
                        chosen = p
                        break
                assignment.append(chosen)
            scenarios.append(tuple(assignment))
        return scenarios

    def _simulate(
        self,
        first_item: int,
        bot_policies: tuple[int, ...],
        own_policy: int,
    ) -> int:
        masks = self.remaining_masks.copy()
        self._remove(first_item, masks)
        masses = self.player_mass.copy()
        values = self.player_value.copy()
        masses[0] += self.m[first_item]
        values[0] += self.v[first_item]

        # If our bag reaches capacity, no opponent can finish heavier than us.
        # Since the win condition is strict M_1 < M_max, this candidate scores 0.
        if masses[0] >= self.capacity:
            return 0

        # Remaining opponents act in the current round; we already acted first.
        for j in range(1, self.k):
            item = self._pick(bot_policies[j - 1], masks, self.capacity - masses[j])
            if item >= 0:
                masses[j] += self.m[item]
                self._remove(item, masks)

        # Future rounds: our move first, then all opponents.
        for _round in range(self.round_no + 1, self.rounds):
            changed = False
            item = self._pick(own_policy, masks, self.capacity - masses[0])
            if item >= 0:
                masses[0] += self.m[item]
                values[0] += self.v[item]
                self._remove(item, masks)
                changed = True
                if masses[0] >= self.capacity:
                    return 0

            for j in range(1, self.k):
                item = self._pick(bot_policies[j - 1], masks, self.capacity - masses[j])
                if item >= 0:
                    masses[j] += self.m[item]
                    self._remove(item, masks)
                    changed = True

            if not changed:
                break

        if masses[0] < max(masses[1:]):
            return values[0]
        return 0

    def choose(self) -> int:
        """Return a zero-based item index, or -1 only if no item fits."""
        candidates = [
            i for i in range(self.n)
            if (self.available_mask >> i) & 1
            and self.player_mass[0] + self.m[i] <= self.capacity
        ]
        if not candidates:
            return -1

        # If all legal moves fill our bag, all of them guarantee score zero.
        safe = [i for i in candidates if self.player_mass[0] + self.m[i] < self.capacity]
        if safe:
            candidates = safe

        # Evaluate promising candidates first, so a time cutoff still has a fallback.
        candidates.sort(
            key=lambda i: (self.v[i] / math.sqrt(self.m[i]), self.v[i], -self.m[i]),
            reverse=True,
        )
        scenarios = self._make_scenarios()
        now = time.process_time()
        remaining_time = max(0.0, self.cpu_deadline - now)
        turns_left = max(1, self.rounds - self.round_no)
        turn_deadline = now + 0.92 * remaining_time / turns_left

        best_item = candidates[0]
        best_score = -1.0
        evaluated_any = False

        for item in candidates:
            if time.process_time() >= turn_deadline:
                break

            policy_means = []
            for own_policy in self.own_policies:
                total = 0.0
                for scenario_index, bot_policies in enumerate(scenarios):
                    total += self._simulate(
                        item,
                        bot_policies,
                        own_policy,
                    )
                policy_means.append(total / len(scenarios))

            policy_means.sort(reverse=True)
            # Best continuation plus a small second-best contribution reduces
            # over-reliance on one brittle future policy.
            score = 0.82 * policy_means[0] + 0.18 * policy_means[1]
            score += 1e-7 * self.v[item] - 1e-9 * self.m[item]

            if score > best_score:
                best_score = score
                best_item = item
                evaluated_any = True

        # If the deadline is reached before the first complete evaluation, keep
        # the deterministic value-density fallback selected above.
        return best_item if evaluated_any else candidates[0]

    def observe(self, actions: Iterable[int]) -> None:
        """Consume K-1 one-based actions; 0 means a forced pass."""
        action_list = list(actions)
        if len(action_list) != self.k - 1:
            raise ValueError(f"expected {self.k - 1} opponent actions, got {len(action_list)}")

        for j, action in enumerate(action_list, start=1):
            if action == 0:
                # A pass is legal only when no remaining item fits; no model update needed.
                continue
            actual = action - 1
            if not (0 <= actual < self.n):
                raise ValueError(f"invalid opponent item id: {action}")
            if not ((self.available_mask >> actual) & 1):
                raise ValueError(f"opponent selected unavailable item: {action}")
            free_capacity = self.capacity - self.player_mass[j]
            self._update_posterior(j - 1, actual, free_capacity)
            self.available_mask &= ~(1 << actual)
            self._remove(actual, self.remaining_masks)
            self.player_mass[j] += self.m[actual]
            self.player_value[j] += self.v[actual]

        self.round_no += 1

    def commit_own_action(self, item_index: int) -> None:
        if item_index < 0:
            if any(
                self.m[i] <= self.capacity - self.player_mass[0]
                for i in range(self.n)
                if (self.available_mask >> i) & 1
            ):
                raise ValueError("cannot pass while a fitting item remains")
            return
        if not (0 <= item_index < self.n):
            raise ValueError("invalid own item index")
        if not ((self.available_mask >> item_index) & 1):
            raise ValueError("own item is unavailable")
        if self.player_mass[0] + self.m[item_index] > self.capacity:
            raise ValueError("own item exceeds capacity")
        self.available_mask &= ~(1 << item_index)
        self._remove(item_index, self.remaining_masks)
        self.player_mass[0] += self.m[item_index]
        self.player_value[0] += self.v[item_index]


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
        if len(actions) != k - 1:
            return
        try:
            agent.observe(actions)
        except ValueError:
            return


if __name__ == "__main__":
    solve()
