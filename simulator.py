"""Offline deterministic simulator for the Greedy Dwarves interactive protocol."""
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
    opponent_policies: list[int],
    own_mode: str = "agent",
    own_policy: int = 1,
    scenario_count: int = 8,
) -> GameResult:
    k, n, rounds, capacity, masses, values = instance
    agent = GreedyDwarvesAgent(
        k, n, rounds, capacity, masses, values,
        scenario_count=scenario_count,
        cpu_budget=100.0,
    )

    for _ in range(rounds):
        if own_mode == "agent":
            own_item = agent.choose()
        else:
            own_item = agent._pick(own_policy, agent.remaining_masks, capacity - agent.player_mass[0])
        agent.commit_own_action(own_item)

        # Construct the complete response line using a separate simulated state.
        sim_masks = agent.remaining_masks.copy()
        sim_masses = agent.player_mass.copy()
        actions = []
        for j in range(1, k):
            policy = opponent_policies[j - 1]
            item = agent._pick(policy, sim_masks, capacity - sim_masses[j])
            if item >= 0:
                sim_masses[j] += masses[item]
                agent._remove(item, sim_masks)
                actions.append(item + 1)
            else:
                actions.append(0)
        agent.observe(actions)

    own_mass = agent.player_mass[0]
    other_max = max(agent.player_mass[1:])
    score = agent.player_value[0] if own_mass < other_max else 0
    return GameResult(
        score=score,
        own_mass=own_mass,
        max_other_mass=other_max,
        own_value=agent.player_value[0],
        opponent_masses=agent.player_mass[1:],
        opponent_values=agent.player_value[1:],
    )


def random_opponents(seed: int, k: int, model_count: int = 8) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(model_count) for _ in range(k - 1)]
