"""Самодостаточное случайное решение для текстового протокола v3.

Запуск: python starter.py [--seed 42]. Весь stdout занят протоколом.
Для своей стратегии замените choose_request(); остальной код читает и
восстанавливает состояние. Игрок участника всегда имеет номер 0.
"""

import argparse
import random
import sys
from dataclasses import dataclass


def read_line():
    line = sys.stdin.readline()
    if line == "":
        raise EOFError("Интерактор закрыл вход до конца партии")
    return line.strip()


def scaled_decimal(token, places):
    """Точная десятичная запись -> целое число в долях 10**(-places)."""
    whole, _, fraction = token.partition(".")
    if len(fraction) > places:
        raise ValueError("Неожиданная точность ресурса")
    return int(whole) * 10**places + int(fraction.ljust(places, "0") or "0")


@dataclass
class Game:
    width: int
    height: int
    players: int
    turns: int
    places: int
    scale: int
    a: int
    b: int
    stations: list
    blocked: set


def read_initial():
    width, height = map(int, read_line().split())
    players = int(read_line())
    turns = int(read_line())
    initial = read_line()
    a, b = read_line().split()
    places = max(len(s.partition(".")[2]) for s in (initial, a, b))
    obstacles = {tuple(map(int, read_line().split())) for _ in range(int(read_line()))}
    stations = [tuple(map(int, read_line().split())) for _ in range(players)]
    game = Game(width, height, players, turns, places, 10**places,
                scaled_decimal(a, places), scaled_decimal(b, places),
                stations, obstacles | set(stations))
    resources = [scaled_decimal(initial, places)] * players
    robots = [{} for _ in range(players)]
    return game, resources, robots


def available_cells(game, robots, player):
    """Соседи станции и старых роботов, кроме препятствий и всех станций."""
    cells = set()
    for x, y in [game.stations[player], *robots[player]]:
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (0 <= nx < game.width and 0 <= ny < game.height
                    and (nx, ny) not in game.blocked):
                cells.add((nx, ny))
    return cells


def choose_request(game, turn, resources, robots, rng):
    """Здесь можно написать свою стратегию. Ресурсы масштабированы, роботы нет."""
    cells = sorted(available_cells(game, robots, 0))
    budget = resources[0] // game.scale
    if not cells or budget == 0:
        return {}

    # Сначала случайное общее число роботов, включая возможный пропуск.
    remaining = rng.randint(0, budget)
    rng.shuffle(cells)
    request = {}
    # Случайное распределение не обязано быть равномерным по всем заявкам.
    # Цикл идёт по клеткам, поэтому большой запас не замедляет решение.
    for index, cell in enumerate(cells):
        count = remaining if index == len(cells) - 1 else rng.randint(0, remaining)
        if count:
            request[cell] = count
        remaining -= count
    return request


def read_request():
    tokens = list(map(int, read_line().split()))
    if len(tokens) % 3:
        raise ValueError("Неполная тройка в заявке")
    return {(tokens[i], tokens[i + 1]): tokens[i + 2]
            for i in range(0, len(tokens), 3)}


def apply_requests(game, robots, requests):
    # Все множества доступности вычисляются ДО установки новых роботов.
    available = [available_cells(game, robots, p) for p in range(game.players)]
    for player, request in enumerate(requests):
        for cell, count in request.items():
            if cell in available[player]:
                robots[player][cell] = robots[player].get(cell, 0) + count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="фиксировать случайность для отладки")
    rng = random.Random(parser.parse_args().seed)
    game, resources, robots = read_initial()
    for turn in range(1, game.turns + 1):
        own = choose_request(game, turn, resources, robots, rng)
        line = " ".join(f"{x} {y} {count}" for (x, y), count in own.items())
        # Запись байтов гарантирует LF в том числе на Windows; пустая строка — пропуск.
        sys.stdout.buffer.write((line + "\n").encode("ascii"))
        sys.stdout.buffer.flush()

        requests = [own] + [read_request() for _ in range(game.players - 1)]
        resources = [scaled_decimal(s, game.places) for s in read_line().split()]
        if len(resources) != game.players:
            raise ValueError("Неверное число запасов в ответе")
        apply_requests(game, robots, requests)
    # Последний ответ, включая итоговые ресурсы, прочитан. Больше ничего не выводим.


if __name__ == "__main__":
    main()
