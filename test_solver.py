import math
import unittest

from solution import GreedyDwarvesAgent
from simulator import make_instance, random_shared_policy, run_game


class SolverTests(unittest.TestCase):
    def test_forced_pass_when_no_item_fits(self):
        agent = GreedyDwarvesAgent(2, 3, 5, 10, [11, 12, 13], [10, 20, 30])
        self.assertEqual(agent.choose(), -1)
        agent.commit_own_action(-1)
        agent.observe([0])
        self.assertEqual(agent.round_no, 1)

    def test_choose_and_commit_are_legal(self):
        for seed in range(30):
            k, n, t, w, m, v = make_instance(seed, k=4, n=40, rounds=8, capacity=30)
            agent = GreedyDwarvesAgent(k, n, t, w, m, v, cpu_budget=100)
            used = set()
            for _ in range(t):
                item = agent.choose()
                if item >= 0:
                    self.assertNotIn(item, used)
                    self.assertLessEqual(agent.player_mass[0] + m[item], w)
                    used.add(item)
                agent.commit_own_action(item)
                agent.observe([0] * (k - 1))

    def test_full_simulations_never_break_capacity(self):
        for seed in range(20):
            instance = make_instance(seed, k=5, n=50, rounds=12, capacity=32)
            policy = random_shared_policy(seed + 1000)
            result = run_game(instance, policy)
            self.assertLessEqual(result.own_mass, instance[3])
            self.assertTrue(all(x <= instance[3] for x in result.opponent_masses))
            self.assertGreaterEqual(result.score, 0)

    def test_one_shared_posterior(self):
        agent = GreedyDwarvesAgent(
            3, 8, 5, 10,
            [7, 7, 1, 5, 9, 8, 7, 5],
            [62, 46, 75, 28, 65, 18, 37, 18],
            cpu_budget=100,
        )
        self.assertEqual(len(agent.posterior), agent.p_count)
        agent.commit_own_action(0)
        before = agent.posterior[:]
        agent.observe([2, 4])
        self.assertNotEqual(agent.posterior, before)
        self.assertAlmostEqual(sum(agent.posterior), 1.0, places=9)
        self.assertTrue(all(x > 0 for x in agent.posterior))


if __name__ == "__main__":
    unittest.main()
