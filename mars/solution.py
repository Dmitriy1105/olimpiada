#!/usr/bin/env python3
"""Mars settlement agent — Python 3.11, interactive protocol v3.

v2 policy:
- exact marginal production over the remaining horizon;
- a one-time bonus for opening new frontier cells;
- ties favour cells in an opponent's current frontier, then cells with
  more free neighbours;
- a heap merges per-cell diminishing-return sequences efficiently.
Only the Python standard library is required.
"""
from __future__ import annotations

import heapq
import sys
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Set, Tuple

Cell = Tuple[int, int]
Request = Dict[Cell, int]


def scaled_decimal(token: str, places: int) -> int:
    whole, _, fraction = token.partition(".")
    if len(fraction) > places:
        raise ValueError("decimal precision exceeds initial scale")
    return int(whole) * (10 ** places) + int(fraction.ljust(places, "0") or "0")


def parse_request(line: str) -> Request:
    tokens = line.split()
    if not tokens:
        return {}
    if len(tokens) % 3:
        raise ValueError("incomplete request triple")
    result: Request = {}
    for i in range(0, len(tokens), 3):
        x, y, count = int(tokens[i]), int(tokens[i + 1]), int(tokens[i + 2])
        cell = (x, y)
        if count <= 0 or cell in result:
            raise ValueError("invalid or duplicate request cell")
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
    def __init__(
        self,
        initial_data,
        expansion_weight_num: int = 12,
        tie_degree_num: int = 4,
    ):
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
        self.expansion_weight_num = expansion_weight_num
        self.tie_degree_num = tie_degree_num

        # Stations are blocked as targets but remain valid network sources.
        self._neighbours: Dict[Cell, List[Cell]] = {}
        for y in range(self.height):
            for x in range(self.width):
                cell = (x, y)
                self._neighbours[cell] = [
                    nxt for nxt in (
                        (x - 1, y), (x + 1, y),
                        (x, y - 1), (x, y + 1),
                    )
                    if 0 <= nxt[0] < self.width
                    and 0 <= nxt[1] < self.height
                    and nxt not in self.blocked
                ]

    def neighbors(self, cell: Cell) -> Iterable[Cell]:
        return self._neighbours.get(cell, ())

    def available_cells(self, player: int) -> Set[Cell]:
        cells: Set[Cell] = set()
        for source in [self.stations[player], *self.robots[player].keys()]:
            cells.update(self.neighbors(source))
        return cells

    def distance_from_network(self, player: int) -> Dict[Cell, int]:
        """Minimum future placement delay; current frontier has distance 1."""
        distance: Dict[Cell, int] = {}
        queue = deque()
        for source in [self.stations[player], *self.robots[player].keys()]:
            for cell in self.neighbors(source):
                if cell not in distance:
                    distance[cell] = 1
                    queue.append(cell)

        while queue:
            cell = queue.popleft()
            next_distance = distance[cell] + 1
            for nxt in self.neighbors(cell):
                if nxt not in distance:
                    distance[nxt] = next_distance
                    queue.append(nxt)
        return distance

    def _income(self, own_count: int, enemy_count: int) -> int:
        total = own_count + enemy_count
        return own_count * max(0, self.a - self.b * total)

    def choose_request(self) -> Request:
        remaining_turns = self.turns - self.turn
        available = self.available_cells(0)
        budget = self.resources[0] // self.scale
        if remaining_turns <= 0 or budget <= 0 or not available:
            return {}

        own_distance = self.distance_from_network(0)
        enemy_distances = [
            self.distance_from_network(player)
            for player in range(1, self.players)
        ]
        enemy_total = {
            cell: sum(
                self.robots[player].get(cell, 0)
                for player in range(1, self.players)
            )
            for cell in available
        }

        # On equal economic value, claim the shared frontier first.
        tie_priority: Dict[Cell, int] = {}
        for cell in available:
            nearest_enemy = min(
                (dist.get(cell, 10**6) for dist in enemy_distances),
                default=10**6,
            )
            margin = nearest_enemy - own_distance.get(cell, 1)
            degree = len(self.neighbors(cell))
            tie_priority[cell] = -100 * margin + self.tie_degree_num * degree

        heap = []
        for cell in available:
            own_count = self.robots[0].get(cell, 0)
            enemy_count = enemy_total[cell]
            delta = (
                self._income(own_count + 1, enemy_count)
                - self._income(own_count, enemy_count)
            )
            if delta <= 0:
                continue

            bonus = 0
            if own_count == 0 and remaining_turns > 1:
                fresh = sum(
                    1 for nxt in self.neighbors(cell)
                    if nxt not in available
                )
                bonus = (
                    fresh * (remaining_turns - 1)
                    * (self.a - self.b) * self.expansion_weight_num
                ) // 100

            net_value = delta * remaining_turns + bonus - self.scale
            if net_value > 0:
                x, y = cell
                heapq.heappush(
                    heap,
                    (-net_value, -tie_priority[cell], x, y, cell, own_count),
                )

        request: Dict[Cell, int] = defaultdict(int)

        # Each cell has diminishing marginal returns. The heap merges these
        # per-cell sequences without rescanning the whole frontier per robot.
        while budget > 0 and heap:
            neg_value, neg_tie, x, y, cell, own_count = heapq.heappop(heap)
            if -neg_value <= 0:
                break

            request[cell] += 1
            budget -= 1
            own_count += 1

            enemy_count = enemy_total[cell]
            delta = (
                self._income(own_count + 1, enemy_count)
                - self._income(own_count, enemy_count)
            )
            net_value = delta * remaining_turns - self.scale
            if net_value > 0:
                heapq.heappush(
                    heap,
                    (-net_value, neg_tie, x, y, cell, own_count),
                )

        return dict(request)

    def update(
        self,
        own_request: Request,
        opponent_requests: List[Request],
        resource_line: str,
    ) -> None:
        requests = [own_request] + opponent_requests
        if len(requests) != self.players:
            raise ValueError("wrong number of player requests")

        # Accessibility is frozen at the start of the turn for every player.
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
        agent = MarsAgent(read_initial())
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
                    raise EOFError("EOF while reading opponent request")
                opponent_requests.append(
                    parse_request(raw.decode("ascii").strip())
                )

            raw_resources = sys.stdin.buffer.readline()
            if raw_resources == b"":
                raise EOFError("EOF while reading resource line")
            agent.update(
                request,
                opponent_requests,
                raw_resources.decode("ascii").strip(),
            )
        return 0
    except (EOFError, ValueError, UnicodeDecodeError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
