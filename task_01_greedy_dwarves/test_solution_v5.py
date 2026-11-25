import random

from solution_v5 import GreedyDwarvesAgent


def reference_sim(agent, first_item, bot_policy, own_policy):
    available = {
        i for i in range(agent.N)
        if (agent.available_mask >> i) & 1
    }
    masses = agent.masses.copy()
    values = agent.values.copy()
    available.remove(first_item)
    masses[0] += agent.m[first_item]
    values[0] += agent.v[first_item]

    for r in range(agent.round_no, agent.T):
        start = 1 if r == agent.round_no else 0
        changed = False
        for j in range(start, agent.K):
            order = agent.orders[own_policy] if j == 0 else agent.orders[bot_policy]
            chosen = -1
            for item in order:
                if item in available and agent.m[item] <= agent.W - masses[j]:
                    chosen = item
                    break
            if chosen >= 0:
                available.remove(chosen)
                masses[j] += agent.m[chosen]
                values[j] += agent.v[chosen]
                changed = True

        if not available:
            break
        # The current round starts after our move, so no bot action in this
        # round does not imply that we cannot take an item in a later round.
        if r > agent.round_no and not changed:
            break

    return values[0] if masses[0] < max(masses[1:]) else 0


def test_simulator_matches_set_reference():
    rng = random.Random(20261125)
    for _ in range(300):
        n = rng.randint(10, 40)
        k = rng.randint(2, 8)
        t = rng.randint(5, 15)
        w = rng.randint(10, 50)
        masses = [rng.randint(1, 50) for _ in range(n)]
        values = [rng.randint(1, 100) for _ in range(n)]
        agent = GreedyDwarvesAgent(k, n, t, w, masses, values)

        removed = set(rng.sample(range(n), rng.randint(0, min(10, n - 1))))
        for item in removed:
            agent.available_mask &= ~(1 << item)
            for p in range(agent.P):
                agent.remaining_masks[p] &= ~agent.item_bits[p][item]

        agent.masses = [rng.randint(0, w) for _ in range(k)]
        agent.values = [rng.randint(0, 300) for _ in range(k)]
        agent.round_no = rng.randrange(t)

        candidates = [
            i for i in range(n)
            if ((agent.available_mask >> i) & 1)
            and agent.m[i] + agent.masses[0] <= w
        ]
        for item in candidates:
            for bp in range(agent.P):
                for op in agent.own_policies:
                    got = agent._simulate(item, bp, op)
                    expected = reference_sim(agent, item, bp, op)
                    assert got == expected, (got, expected)


def test_random_shared_policy_games():
    rng = random.Random(20261126)
    for _ in range(100):
        k = rng.randint(2, 10)
        n = rng.randint(10, 100)
        t = rng.randint(5, 20)
        w = rng.randint(10, 50)
        masses = [rng.randint(1, 50) for _ in range(n)]
        values = [rng.randint(1, 100) for _ in range(n)]
        agent = GreedyDwarvesAgent(k, n, t, w, masses, values)
        hidden = rng.randrange(agent.P)

        for _round in range(t):
            chosen = agent.choose()
            if chosen < 0:
                assert not any(
                    ((agent.available_mask >> i) & 1)
                    and masses[i] <= w - agent.masses[0]
                    for i in range(n)
                )
            else:
                assert (agent.available_mask >> chosen) & 1
                assert agent.masses[0] + masses[chosen] <= w

            agent.commit_own_action(chosen)
            bot_mask = agent.remaining_masks[hidden]
            local_masses = agent.masses.copy()
            actions = []
            for j in range(1, k):
                item = agent._pick(hidden, bot_mask, w - local_masses[j])
                if item >= 0:
                    actions.append(item + 1)
                    bot_mask &= ~agent.item_bits[hidden][item]
                    local_masses[j] += masses[item]
                else:
                    actions.append(0)

            agent.observe(actions)
            assert all(0 <= x <= w for x in agent.masses)
            assert abs(sum(agent.posterior) - 1.0) < 1e-9
            assert all(x >= 0.0 for x in agent.posterior)


if __name__ == "__main__":
    test_simulator_matches_set_reference()
    test_random_shared_policy_games()
    print("V5 tests: PASS (300 simulator comparisons + 100 randomized games)")
