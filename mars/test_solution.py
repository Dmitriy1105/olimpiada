import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location(
    "mars_simulator", ROOT / "simulate.py"
)
simulator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simulator)

spec = importlib.util.spec_from_file_location(
    "mars_solution", ROOT / "solution.py"
)
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)


def make_agent_initial(params):
    width, height, players, turns, initial, a, b, obstacles, stations, _ = params
    places = max(
        -value.as_tuple().exponent for value in (initial, a, b)
    )
    scale = 10 ** places
    return (
        width, height, players, turns, places, scale,
        int(initial * scale), int(a * scale), int(b * scale),
        obstacles, stations,
    )


def clip_request(request, budget):
    result = {}
    for cell, count in sorted(request.items()):
        take = min(count, budget)
        if take:
            result[cell] = take
            budget -= take
        if budget == 0:
            break
    return result


class MarsTests(unittest.TestCase):
    def test_all_ten_histories_reproduce_exact_resources(self):
        reports = simulator.validate_histories(ROOT)
        self.assertEqual(len(reports), 10)
        self.assertEqual(
            [row["strategy_id"] for row in reports],
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        )

    def test_agent_requests_are_legal_on_all_open_maps(self):
        for index in range(1, 11):
            with self.subTest(test=index):
                params = simulator.parse_test_map(ROOT / "tests" / f"{index:02d}")
                game = simulator.Game(params)
                agent = solution.MarsAgent(make_agent_initial(params))
                history = (
                    ROOT / "histories" / f"{index:02d}.txt"
                ).read_text(encoding="ascii").splitlines()

                for turn in range(game.turns):
                    own = agent.choose_request()
                    self.assertLessEqual(
                        sum(own.values()),
                        game.resources[0] // game.scale,
                    )
                    self.assertTrue(set(own).issubset(game.available(0)))

                    opponent = simulator.parse_request(
                        history[3 * turn + 1]
                    )
                    opponent = clip_request(
                        opponent, game.resources[1] // game.scale
                    )

                    resources = game.step([own, opponent])
                    resource_line = " ".join(game.resource_tokens())
                    agent.update(own, [opponent], resource_line)
                    self.assertEqual(agent.resources, resources)
                    self.assertEqual(
                        dict(agent.robots[0]), dict(game.robots[0])
                    )
                    self.assertEqual(
                        dict(agent.robots[1]), dict(game.robots[1])
                    )

    def test_interactive_protocol_on_test_01_and_10(self):
        for index in (1, 10):
            with self.subTest(test=index):
                test_path = ROOT / "tests" / f"{index:02d}"
                params = simulator.parse_test_map(test_path)
                game = simulator.Game(params)
                test_lines = test_path.read_text(
                    encoding="ascii"
                ).splitlines()
                # The final token is the open-test opponent strategy ID,
                # not part of the interactive initial block.
                initial_block = "\n".join(test_lines[:-1]) + "\n"
                history = (
                    ROOT / "histories" / f"{index:02d}.txt"
                ).read_text(encoding="ascii").splitlines()

                process = subprocess.Popen(
                    [sys.executable, str(ROOT / "solution.py")],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                process.stdin.write(initial_block)
                process.stdin.flush()

                output_lines = []
                try:
                    for turn in range(game.turns):
                        own_line = process.stdout.readline()
                        self.assertNotEqual(
                            own_line, "",
                            f"agent exited before turn {turn + 1}",
                        )
                        output_lines.append(own_line.rstrip("\n"))
                        own = simulator.parse_request(own_line)
                        self.assertLessEqual(
                            sum(own.values()),
                            game.resources[0] // game.scale,
                        )
                        self.assertTrue(set(own).issubset(game.available(0)))

                        opponent = simulator.parse_request(
                            history[3 * turn + 1]
                        )
                        self.assertLessEqual(
                            sum(opponent.values()),
                            game.resources[1] // game.scale,
                        )
                        game.step([own, opponent])

                        process.stdin.write(
                            history[3 * turn + 1] + "\n"
                        )
                        process.stdin.write(
                            " ".join(game.resource_tokens()) + "\n"
                        )
                        process.stdin.flush()

                    process.stdin.close()
                    return_code = process.wait(timeout=5)
                    remaining_stdout = process.stdout.read()
                    stderr = process.stderr.read()
                    self.assertEqual(return_code, 0, stderr)
                    self.assertEqual(remaining_stdout, "")
                    self.assertEqual(len(output_lines), game.turns)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    if process.stdin and not process.stdin.closed:
                        process.stdin.close()
                    if process.stdout and not process.stdout.closed:
                        process.stdout.close()
                    if process.stderr and not process.stderr.closed:
                        process.stderr.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
