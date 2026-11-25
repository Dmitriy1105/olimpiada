import random
import subprocess
import sys
import unittest
from collections import deque
from pathlib import Path


SOLVER = Path(__file__).with_name("millions_of_paths_v3.py")


def reference_distances(rows, sx, sy):
    h = len(rows)
    w = len(rows[0])
    start = sy * w + sx
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        x, y = u % w, u // w
        nd = dist[u] + 1
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and rows[ny][nx] == ".":
                v = ny * w + nx
                if v not in dist:
                    dist[v] = nd
                    q.append(v)
    return dist


def run_interactive(rows, queries):
    h = len(rows)
    w = len(rows[0])
    p = subprocess.Popen(
        [sys.executable, str(SOLVER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert p.stdin is not None and p.stdout is not None
    p.stdin.write(f"{w} {h}\n")
    p.stdin.write("\n".join(rows) + "\n")
    p.stdin.flush()

    answers = []
    for sx, sy, tx, ty in queries:
        p.stdin.write(f"{sx} {sy} {tx} {ty}\n")
        p.stdin.flush()
        line = p.stdout.readline()
        if not line:
            err = p.stderr.read() if p.stderr else ""
            raise AssertionError(f"solver did not answer query; stderr={err}")
        answers.append(int(line))

    p.stdin.close()
    rc = p.wait(timeout=5)
    err = p.stderr.read() if p.stderr else ""
    extra = p.stdout.read()
    if rc != 0:
        raise AssertionError(f"exit={rc}, stderr={err}")
    if extra:
        raise AssertionError(f"unexpected extra output: {extra!r}")
    return answers


class MillionsOfPathsTests(unittest.TestCase):
    def test_official_samples(self):
        rows1 = ["....", ".##.", "...."]
        q1 = [(0, 0, 3, 2), (1, 0, 1, 2), (3, 2, 3, 2)]
        self.assertEqual(run_interactive(rows1, q1), [5, 4, 0])

        rows2 = ["..#..", "..#..", "..#.."]
        q2 = [(0, 0, 4, 2), (0, 0, 1, 2), (3, 1, 3, 1)]
        self.assertEqual(run_interactive(rows2, q2), [-1, 3, 0])

    def test_random_maps_all_pairs_against_bfs(self):
        rng = random.Random(20261125)
        for case in range(15):
            w = rng.randint(3, 11)
            h = rng.randint(3, 11)
            rows = [
                "".join("." if rng.random() < 0.68 else "#" for _ in range(w))
                for _ in range(h)
            ]
            free = [(x, y) for y in range(h) for x in range(w) if rows[y][x] == "."]
            if len(free) < 2:
                rows = ["." * w for _ in range(h)]
                free = [(x, y) for y in range(h) for x in range(w)]

            queries = [(sx, sy, tx, ty) for sx, sy in free for tx, ty in free]
            expected = []
            cache = {}
            for sx, sy, tx, ty in queries:
                key = (sx, sy)
                if key not in cache:
                    cache[key] = reference_distances(rows, sx, sy)
                expected.append(cache[key].get(ty * w + tx, -1))

            actual = run_interactive(rows, queries)
            self.assertEqual(actual, expected, msg=f"random map #{case}: {rows}")

    def test_vertical_and_horizontal_bitset_paths(self):
        # Tall/narrow rectangles exercise the column bitset branch.
        rows = [
            "...",
            ".#.",
            ".#.",
            "...",
            "...",
            ".#.",
            "...",
            "...",
            "...",
            "...",
        ]
        q = [(0, 0, 1, 9), (2, 0, 1, 9), (0, 9, 2, 0)]
        expected = []
        for sx, sy, tx, ty in q:
            d = reference_distances(rows, sx, sy)
            expected.append(d.get(ty * len(rows[0]) + tx, -1))
        self.assertEqual(run_interactive(rows, q), expected)

        # Wide/short rectangles exercise the row bitset branch.
        rows = [
            "...........",
            "###...#####",
            "...........",
        ]
        q = [(0, 0, 10, 2), (10, 0, 0, 2), (0, 2, 10, 0)]
        expected = []
        for sx, sy, tx, ty in q:
            d = reference_distances(rows, sx, sy)
            expected.append(d.get(ty * len(rows[0]) + tx, -1))
        self.assertEqual(run_interactive(rows, q), expected)


if __name__ == "__main__":
    unittest.main()
