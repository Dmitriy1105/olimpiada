import random
import subprocess
import sys
import unittest
from collections import deque
from pathlib import Path

SOLVER = Path(__file__).with_name("millions_of_paths_v4.py")


def bfs_distances(rows, sx, sy):
    h = len(rows)
    w = len(rows[0])
    start = sy * w + sx
    dist = {start: 0}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        x, y = u % w, u // w
        nd = dist[u] + 1
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and rows[ny][nx] == ".":
                v = ny * w + nx
                if v not in dist:
                    dist[v] = nd
                    queue.append(v)
    return dist


def run_solver(rows, queries):
    h = len(rows)
    w = len(rows[0])
    proc = subprocess.Popen(
        [sys.executable, str(SOLVER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(f"{w} {h}\n")
    proc.stdin.write("\n".join(rows) + "\n")
    proc.stdin.flush()

    answers = []
    for query in queries:
        proc.stdin.write("{} {} {} {}\n".format(*query))
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            error = proc.stderr.read() if proc.stderr else ""
            raise AssertionError(f"no answer; stderr={error}")
        answers.append(int(line))

    proc.stdin.close()
    code = proc.wait(timeout=10)
    if code:
        error = proc.stderr.read() if proc.stderr else ""
        raise AssertionError(f"exit={code}; stderr={error}")
    proc.stdout.close()
    if proc.stderr:
        proc.stderr.close()
    return answers


class JPSV4Tests(unittest.TestCase):
    def test_official_samples(self):
        self.assertEqual(
            run_solver(["....", ".##.", "...."],
                       [(0, 0, 3, 2), (1, 0, 1, 2), (3, 2, 3, 2)]),
            [5, 4, 0],
        )
        self.assertEqual(
            run_solver(["..#..", "..#..", "..#.."],
                       [(0, 0, 4, 2), (0, 0, 1, 2), (3, 1, 3, 1)]),
            [-1, 3, 0],
        )

    def test_all_pairs_on_random_small_maps(self):
        rng = random.Random(20261125)
        for case in range(30):
            w = rng.randint(3, 10)
            h = rng.randint(3, 10)
            p = rng.choice((0.15, 0.25, 0.35, 0.45, 0.55))
            rows = [
                "".join("." if rng.random() > p else "#" for _ in range(w))
                for _ in range(h)
            ]
            free = [(x, y) for y in range(h) for x in range(w) if rows[y][x] == "."]
            if len(free) < 2:
                rows = ["." * w for _ in range(h)]
                free = [(x, y) for y in range(h) for x in range(w)]

            queries = [(sx, sy, tx, ty) for sx, sy in free for tx, ty in free]
            expected = []
            bfs_cache = {}
            for sx, sy, tx, ty in queries:
                key = (sx, sy)
                if key not in bfs_cache:
                    bfs_cache[key] = bfs_distances(rows, sx, sy)
                expected.append(bfs_cache[key].get(ty * w + tx, -1))

            actual = run_solver(rows, queries)
            self.assertEqual(actual, expected, msg=f"case={case}, map={rows}")

    def test_maze_and_forced_turn_patterns(self):
        maps = [
            [
                "....#....",
                "...##....",
                ".#.......",
                ".########",
                ".........",
            ],
            [
                "...........",
                "###.#######",
                "...#.......",
                ".##.#######",
                "...........",
            ],
            [
                "....",
                "..#.",
                ".#..",
                "....",
            ],
        ]
        for rows in maps:
            free = [(x, y) for y in range(len(rows)) for x in range(len(rows[0])) if rows[y][x] == "."]
            queries = [(sx, sy, tx, ty) for sx, sy in free for tx, ty in free]
            expected = []
            cache = {}
            w = len(rows[0])
            for sx, sy, tx, ty in queries:
                key = (sx, sy)
                if key not in cache:
                    cache[key] = bfs_distances(rows, sx, sy)
                expected.append(cache[key].get(ty * w + tx, -1))
            self.assertEqual(run_solver(rows, queries), expected)


if __name__ == "__main__":
    unittest.main()
