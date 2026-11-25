"""Offline simulator for games where all opponents share one hidden policy."""
from __future__ import annotations

import random
from dataclasses import dataclass

from solution import GreedyDwarvesAgent


@dataclass
class GameResult:
    score: int
    own_mass: int
    max_other_mass: int
    own_value: int
    opponent_masses: list[int]
    opponent_values: list[int]


def make_instance(seed: int, k: int = 5, n: int = 60, rounds: int = 15, capacity: int = 35):
    rng = random.Random(seed)
    masses = [rng.randint(1, 50) for _ in range(n)]
    values = [rng.randint(1, 100) for _ in range(n)]
    return k, n, rounds, capacity, masses, values


def run_game(
    instance,
    opponent_policy: int,
    own_mode: str = "agent",
    own_policy: int = 1,
) -> GameResult:
    k, n, rounds, capacity, masses, values = instance
    agent = GreedyDwarvesAgent(
        k, n, rounds, capacity, masses, values, cpu_budget=100.0
    )

    for _ in range(rounds):
        if own_mode == "agent":
            own_item = agent.choose()
        else:
            own_item = agent._pick(
                own_policy,
                agent.remaining_masks[own_policy],
                capacity - agent.player_mass[0],
            )
        agent.commit_own_action(own_item)

        # All opponents use the same policy, but each sees the updated state
        # after our move and all preceding opponents' moves.
        bot_mask = agent.remaining_masks[opponent_policy]
        actions = []
        for j in range(1, k):
            item = agent._pick(
                opponent_policy,
                bot_mask,
                capacity - agent.player_mass[j],
            )
            if item >= 0:
                bot_mask &= ~agent.item_bits[opponent_policy][item]
                actions.append(item + 1)
            else:
                actions.append(0)

        agent.observe(actions)

    own_mass = agent.player_mass[0]
    max_other = max(agent.player_mass[1:])
    score = agent.player_value[0] if own_mass < max_other else 0
    return GameResult(
        score=score,
        own_mass=own_mass,
        max_other_mass=max_other,
        own_value=agent.player_value[0],
        opponent_masses=agent.player_mass[1:],
        opponent_values=agent.player_value[1:],
    )


def random_shared_policy(seed: int, policy_count: int = 10) -> int:
    """Choose one hidden policy for the entire test, not one per opponent."""
    return random.Random(seed).randrange(policy_count)
