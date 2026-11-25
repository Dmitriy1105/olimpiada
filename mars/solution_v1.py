#!/usr/bin/env python3
"""Interactive agent for "Освоение Марса" (protocol v3), Python 3.11.

The agent uses exact fixed-point arithmetic for game resources and robot
counts. Its placement policy greedily allocates robots by their marginal
long-term production, with a small bonus for opening valuable frontier cells.
Only the standard library is required.
"""

from __future__ import annotations

import heapq
import math
import sys
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Set, Tuple

Cell = Tuple[int, int]
Request = Dict[Cell, int]


def scaled_decimal(token: str, places: int) -> int:
    """Convert a non-negative finite decimal to an integer at 10**(-places)."""
    whole, dot, fraction = token.partition(".")
    if len(fraction) > places:
        raise ValueError("decimal precision exceeds the initial scale")
    return int(whole) * (10 ** places) + int(fraction.ljust(places, "0") or "0")


def parse_request(line: str) -> Request:
    values = line.split()
    if not values:
        return {}
    if len(values) % 3:
        raise ValueError("incomplete request triple")
    result: Request = {}
    for i in range(0, len(values), 3):
        x, y, count = int(values[i]), int(values[i + 1]), int(values[i + 2])
        if count <= 0:
            raise ValueError("request counts must be positive")
        cell = (x, y)
        if cell in result:
            raise ValueError("duplicate cell in request")
        result[cell] = count
    return result


def parse_resource_line(line: str, places: int, players: int) -> List[int]:
    tokens = line.split()
    if len(tokens) != players:
        raise ValueError("wrong number of resource values")
    return [scaled_decimal(token, places) for token in tokens]


def read_initial():
    def read_tokens() -> List[str]:
        raw = sys.stdin.buffer.readline()
        if raw == b"":
            raise EOFError("EOF in initial block")
        return raw.decode("ascii").split()

    width, height = map(int, read_tokens())
    players = int(read_tokens()[0])
    turns = int(read_tokens()[0])
    initial_token = read_tokens()[0]
    a_token, b_token = read_tokens()
    obstacle_count = int(read_tokens()[0])

    obstacles: Set[Cell] = set()
    for _ in range(obstacle_count):
        obstacles.add(tuple(map(int, read_tokens())))

    stations: List[Cell] = []
    for _ in range(players):
        stations.append(tuple(map(int, read_tokens())))

    places = max(
        len(token.partition(".")[2])
        for token in (initial_token, a_token, b_token)
    )
    scale = 10 ** places
    initial = scaled_decimal(initial_token, places)
    a = scaled_decimal(a_token, places)
    b = scaled_decimal(b_token, places)

    return (
        width, height, players, turns, places, scale,
        initial, a, b, obstacles, stations,
    )


class MarsAgent:
    def __init__(self, initial_data, expansion_weight: float = 1.5):
        (
            self.width, self.height, self.players, self.turns,
            self.places, self.scale, initial, self.a, self.b,
            self.obstacles, self.stations,
        ) = initial_data

        self.resources = [initial] * self.players
        self.robots: List[Dict[Cell, int]] = [
            defaultdict(int) for _ in range(self.players)
        ]
        self.blocked = set(self.obstacles)
        self.blocked.update(self.stations)
        self.turn = 0
        self.expansion_weight = expansion_weight

        # Static shortest-path distances are useful for estimating which side
        # of the map is naturally safer for each player.
        self.static_distances = [
            self._distance_from_station(player)
            for player in range(self.players)
        ]
        self.static_margin: Dict[Cell, int] = {}
        for y in range(self.height):
            for x in range(self.width):
                cell = (x, y)
                if cell in self.blocked:
                    continue
                own_distance = self.static_distances[0].get(cell, 10**6)
                enemy_distance = min(
                    (
                        self.static_distances[p].get(cell, 10**6)
                        for p in range(1, self.players)
                    ),
                    default=10**6,
                )
                self.static_margin[cell] = enemy_distance - own_distance

    def _neighbors(self, cell: Cell) -> Iterable[Cell]:
        x, y = cell
        if x > 0:
            yield (x - 1, y)
        if x + 1 < self.width:
            yield (x + 1, y)
        if y > 0:
            yield (x, y - 1)
        if y + 1 < self.height:
            yield (x, y + 1)

    def _distance_from_station(self, player: int) -> Dict[Cell, int]:
        start = self.stations[player]
        distance = {start: 0}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            next_distance = distance[cell] + 1
            for nxt in self._neighbors(cell):
                if nxt in self.blocked or nxt in distance:
                    continue
                distance[nxt] = next_distance
                queue.append(nxt)
        return distance

    def _distance_from_network(self, player: int) -> Dict[Cell, int]:
        """Minimum number of future placements needed to reach each cell."""
        sources = set(self.robots[player])
        sources.add(self.stations[player])
        distance: Dict[Cell, int] = {}
        queue = deque()

        # A cell adjacent to an existing station/robot can be placed this turn.
        for source in sources:
            for cell in self._neighbors(source):
                if cell in self.blocked or cell in distance:
                    continue
                distance[cell] = 1
                queue.append(cell)

        while queue:
            cell = queue.popleft()
            next_distance = distance[cell] + 1
            for nxt in self._neighbors(cell):
                if nxt in self.blocked or nxt in distance:
                    continue
                distance[nxt] = next_distance
                queue.append(nxt)
        return distance

    def available_cells(self, player: int) -> Set[Cell]:
        cells: Set[Cell] = set()
        sources = list(self.robots[player])
        sources.append(self.stations[player])
        for source in sources:
            for cell in self._neighbors(source):
                if cell not in self.blocked:
                    cells.add(cell)
        return cells

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return low if value < low else high if value > high else value

    def _income(self, own_count: int, enemy_count: int) -> int:
        return own_count * max(
            0, self.a - self.b * (own_count + enemy_count)
        )

    def _contest_probability(
        self,
        cell: Cell,
        enemy_distances: List[Dict[Cell, int]],
        enemy_available_count: int,
    ) -> float:
        """Estimate the chance that an opponent adds to this cell later."""
        nearest = min(
            (distance.get(cell, 10**6) for distance in enemy_distances),
            default=10**6,
        )
        margin = self.static_margin.get(cell, 0)

        if nearest >= 10**6:
            probability = 0.02
        else:
            probability = 0.72 * math.exp(
                -0.28 * max(0, nearest - 1)
            )
            if margin > 0:
                probability *= math.exp(-0.16 * margin)
            elif margin < 0:
                probability *= math.exp(min(1.0, 0.22 * -margin))

        if enemy_available_count:
            immediate = min(0.88, 0.42 + 0.16 * (enemy_available_count - 1))
            probability = max(probability, immediate)

        return self._clamp(probability, 0.02, 0.88)

    def choose_request(self) -> Request:
        remaining_turns = self.turns - self.turn
        available = self.available_cells(0)
        if not available:
            return {}

        enemy_available = [
            self.available_cells(player)
            for player in range(1, self.players)
        ]
        # Dynamic distances reflect the actual frontiers, not just stations.
        enemy_distances = [
            self._distance_from_network(player)
            for player in range(1, self.players)
        ]

        def contest_probability(cell: Cell) -> float:
            count = sum(cell in cells for cells in enemy_available)
            return self._contest_probability(
                cell, enemy_distances, count
            )

        # A newly occupied frontier cell opens its currently inaccessible
        # neighbours on the next turn. Divide shared neighbours among the
        # candidate cells so that the same opening is not counted repeatedly.
        coverage = defaultdict(int)
        for cell in available:
            if self.robots[0].get(cell, 0) != 0:
                continue
            for nxt in self._neighbors(cell):
                if nxt not in self.blocked and nxt not in available:
                    coverage[nxt] += 1

        expansion_bonus: Dict[Cell, float] = {}
        for cell in available:
            if self.robots[0].get(cell, 0) > 0:
                expansion_bonus[cell] = 0.0
                continue

            bonus = 0.0
            for nxt in self._neighbors(cell):
                if nxt in self.blocked or nxt in available:
                    continue

                enemy_count = sum(
                    self.robots[p].get(nxt, 0)
                    for p in range(1, self.players)
                )
                probability = contest_probability(nxt)
                first_income = max(
                    0, self.a - self.b * (1 + enemy_count)
                )
                contested_income = max(
                    0, self.a - self.b * (2 + enemy_count)
                )
                expected_income = (
                    (1.0 - probability) * first_income
                    + probability * contested_income
                )

                future_profit = max(
                    0.0,
                    (remaining_turns - 1) * expected_income - self.scale,
                )
                margin = self.static_margin.get(nxt, 0)
                safety = self._clamp(0.55 + 0.08 * margin, 0.20, 1.0)
                degree = sum(
                    neighbour not in self.blocked
                    for neighbour in self._neighbors(nxt)
                )
                branching = 1.0 + 0.08 * max(0, degree - 1)
                future_profit *= safety * branching
                future_profit /= max(1, coverage[nxt])
                bonus += future_profit

            # Small enough not to dominate direct production, but sufficient
            # to prefer useful branches when current marginal yields are close.
            expansion_bonus[cell] = self.expansion_weight * bonus

        budget = self.resources[0] // self.scale
        if budget <= 0:
            return {}

        # Heap entries hold the next robot's marginal value for each cell.
        # All values are expressed in scaled resource units.
        heap = []
        probabilities: Dict[Cell, float] = {}
        for cell in available:
            own_count = self.robots[0].get(cell, 0)
            enemy_count = sum(
                self.robots[p].get(cell, 0)
                for p in range(1, self.players)
            )
            probability = contest_probability(cell)
            probabilities[cell] = probability

            current = (
                (1.0 - probability) * self._income(own_count, enemy_count)
                + probability * self._income(own_count, enemy_count + 1)
            )
            after_one = (
                (1.0 - probability) * self._income(own_count + 1, enemy_count)
                + probability * self._income(own_count + 1, enemy_count + 1)
            )
            marginal = after_one - current

            bonus = expansion_bonus[cell] if own_count == 0 else 0.0
            score = remaining_turns * marginal - self.scale + bonus

            margin = self.static_margin.get(cell, 0)
            degree = sum(
                neighbour not in self.blocked
                for neighbour in self._neighbors(cell)
            )
            tie = 0.10 * margin + 0.04 * degree
            heapq.heappush(heap, (-score, -tie, cell))

        request: Dict[Cell, int] = defaultdict(int)
        while budget > 0 and heap:
            negative_score, negative_tie, cell = heapq.heappop(heap)
            if -negative_score <= 0.0:
                break

            request[cell] += 1
            budget -= 1

            own_count = self.robots[0].get(cell, 0) + request[cell]
            enemy_count = sum(
                self.robots[p].get(cell, 0)
                for p in range(1, self.players)
            )
            probability = probabilities[cell]

            current = (
                (1.0 - probability) * self._income(own_count, enemy_count)
                + probability * self._income(own_count, enemy_count + 1)
            )
            after_one = (
                (1.0 - probability) * self._income(own_count + 1, enemy_count)
                + probability * self._income(own_count + 1, enemy_count + 1)
            )
            marginal = after_one - current

            # The frontier bonus is paid only for the first robot in a cell.
            score = remaining_turns * marginal - self.scale
            heapq.heappush(
                heap, (-score, negative_tie, cell)
            )

        return {
            cell: count
            for cell, count in request.items()
            if count > 0
        }

    def update(
        self,
        own_request: Request,
        opponent_requests: List[Request],
        resource_line: str,
    ) -> None:
        requests = [own_request] + opponent_requests
        if len(requests) != self.players:
            raise ValueError("wrong number of player requests")

        # Availability is evaluated from the snapshot at the beginning
        # of the turn, before any player's new robots are installed.
        available = [
            self.available_cells(player)
            for player in range(self.players)
        ]

        for player, request in enumerate(requests):
            for cell, count in request.items():
                if cell in available[player]:
                    self.robots[player][cell] = (
                        self.robots[player].get(cell, 0) + count
                    )

        self.resources = parse_resource_line(
            resource_line, self.places, self.players
        )
        self.turn += 1


def main() -> int:
    try:
        initial_data = read_initial()
        agent = MarsAgent(initial_data)

        for _ in range(agent.turns):
            request = agent.choose_request()
            line = " ".join(
                f"{x} {y} {count}"
                for (x, y), count in sorted(request.items())
            )
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

            opponent_requests = []
            for _player in range(agent.players - 1):
                raw = sys.stdin.buffer.readline()
                if raw == b"":
                    return 0
                opponent_requests.append(
                    parse_request(raw.decode("ascii").strip())
                )

            raw_resources = sys.stdin.buffer.readline()
            if raw_resources == b"":
                return 0
            resource_line = raw_resources.decode("ascii").strip()
            agent.update(request, opponent_requests, resource_line)

        # The final opponent requests and final resource line have been read.
        # The protocol then closes stdin; no further stdout is permitted.
        return 0
    except (EOFError, ValueError, UnicodeDecodeError):
        # No diagnostics on stdout: any protocol fault is handled by the judge.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
