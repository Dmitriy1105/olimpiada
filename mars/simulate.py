"""Exact local simulator and fixture/history validation for Mars v13."""

from __future__ import annotations

from collections import defaultdict, deque
from decimal import Decimal
from pathlib import Path


def parse_test_map(path):
    lines = Path(path).read_text(encoding="ascii").splitlines()
    width, height = map(int, lines[0].split())
    players = int(lines[1])
    turns = int(lines[2])
    initial = Decimal(lines[3])
    a, b = map(Decimal, lines[4].split())
    obstacle_count = int(lines[5])
    coords = [
        tuple(map(int, line.split()))
        for line in lines[6:6 + obstacle_count + players]
    ]
    obstacles = set(coords[:obstacle_count])
    stations = coords[obstacle_count:obstacle_count + players]
    strategy_id = int(lines[6 + obstacle_count + players])
    return width, height, players, turns, initial, a, b, obstacles, stations, strategy_id


def parse_request(line):
    values = list(map(int, line.split()))
    if len(values) % 3:
        raise ValueError("incomplete request")
    result = {}
    for i in range(0, len(values), 3):
        cell = (values[i], values[i + 1])
        count = values[i + 2]
        if count <= 0 or cell in result:
            raise ValueError("invalid request")
        result[cell] = count
    return result


def scaled(token, places):
    whole, _, fraction = token.partition(".")
    if len(fraction) > places:
        raise ValueError("too many decimal places")
    return int(whole) * 10**places + int(fraction.ljust(places, "0") or "0")


def decimal_places(*values):
    return max(-value.as_tuple().exponent for value in values)


class Game:
    def __init__(self, params):
        (
            self.width, self.height, self.players, self.turns,
            initial, a, b, obstacles, stations, self.strategy_id,
        ) = params
        self.places = decimal_places(initial, a, b)
        self.scale = 10**self.places
        self.initial = int(initial * self.scale)
        self.a = int(a * self.scale)
        self.b = int(b * self.scale)
        self.obstacles = obstacles
        self.stations = stations
        self.blocked = set(obstacles) | set(stations)
        self.resources = [self.initial] * self.players
        self.robots = [defaultdict(int) for _ in range(self.players)]
        self.turn = 0

    def neighbors(self, cell):
        x, y = cell
        if x > 0:
            yield (x - 1, y)
        if x + 1 < self.width:
            yield (x + 1, y)
        if y > 0:
            yield (x, y - 1)
        if y + 1 < self.height:
            yield (x, y + 1)

    def available(self, player):
        result = set()
        sources = [self.stations[player], *self.robots[player].keys()]
        for source in sources:
            for cell in self.neighbors(source):
                if cell not in self.blocked:
                    result.add(cell)
        return result

    def step(self, requests):
        if len(requests) != self.players:
            raise ValueError("wrong number of requests")

        # Snapshot all access sets before installing any new robots.
        available = [self.available(p) for p in range(self.players)]
        costs = [
            sum(request.values()) * self.scale
            for request in requests
        ]
        for p in range(self.players):
            if costs[p] > self.resources[p]:
                raise ValueError(
                    f"player {p} overspent on turn {self.turn + 1}"
                )

        for p in range(self.players):
            self.resources[p] -= costs[p]

        for p, request in enumerate(requests):
            for cell, count in request.items():
                if cell in available[p]:
                    self.robots[p][cell] += count

        # Production is calculated after all requests have been applied.
        for p in range(self.players):
            income = 0
            for cell, own_count in self.robots[p].items():
                total_count = sum(
                    self.robots[q].get(cell, 0)
                    for q in range(self.players)
                )
                income += own_count * max(
                    0, self.a - self.b * total_count
                )
            self.resources[p] += income

        self.turn += 1
        return self.resources[:]

    def resource_tokens(self):
        return [
            format(Decimal(value) / self.scale, "f")
            for value in self.resources
        ]


def validate_histories(root):
    root = Path(root)
    reports = []
    for index in range(1, 11):
        test_path = root / "tests" / f"{index:02d}"
        history_path = root / "histories" / f"{index:02d}.txt"
        params = parse_test_map(test_path)
        game = Game(params)
        lines = history_path.read_text(encoding="ascii").splitlines()
        if len(lines) != 3 * game.turns:
            raise AssertionError(
                f"{index:02d}: expected {3 * game.turns} history lines, got {len(lines)}"
            )

        for turn in range(game.turns):
            requests = [
                parse_request(lines[3 * turn]),
                parse_request(lines[3 * turn + 1]),
            ]
            actual = game.step(requests)
            reported = [
                scaled(token, game.places)
                for token in lines[3 * turn + 2].split()
            ]
            if actual != reported:
                raise AssertionError(
                    f"{index:02d}, turn {turn + 1}: "
                    f"calculated={actual}, reported={reported}"
                )

        reports.append({
            "test": index,
            "strategy_id": params[-1],
            "turns": game.turns,
            "final_resources": [
                float(Decimal(value) / game.scale)
                for value in game.resources
            ],
        })
    return reports


if __name__ == "__main__":
    for report in validate_histories(Path(__file__).resolve().parent):
        print(
            f"test={report['test']:02d} strategy={report['strategy_id']} "
            f"turns={report['turns']} final={report['final_resources']}"
        )
    print("PASS: all 10 histories reproduce exactly")
