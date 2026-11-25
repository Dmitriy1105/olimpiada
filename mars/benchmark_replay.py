"""Exploratory benchmark against fixed recorded opponent actions.

This is NOT an official score: the recorded opponent does not adapt to our
changed actions. If its recorded request exceeds its changed resource budget,
the request is truncated and the number of such turns is reported.
"""

from decimal import Decimal
from pathlib import Path

try:
    from .simulate import Game, parse_test_map, parse_request
    from .solution import MarsAgent
except ImportError:
    from simulate import Game, parse_test_map, parse_request
    from solution import MarsAgent


ROOT = Path(__file__).resolve().parent


def make_agent_initial(params):
    width, height, players, turns, initial, a, b, obstacles, stations, _ = params
    places = max(-v.as_tuple().exponent for v in (initial, a, b))
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
        if take > 0:
            result[cell] = take
            budget -= take
        if budget <= 0:
            break
    return result


def run_one(index):
    params = parse_test_map(ROOT / "tests" / f"{index:02d}")
    game = Game(params)
    agent = MarsAgent(make_agent_initial(params))
    history = (
        ROOT / "histories" / f"{index:02d}.txt"
    ).read_text(encoding="ascii").splitlines()

    clipped_turns = 0
    for turn in range(game.turns):
        own_request = agent.choose_request()
        opponent_request = parse_request(history[3 * turn + 1])
        budget = game.resources[1] // game.scale
        if sum(opponent_request.values()) > budget:
            clipped_turns += 1
            opponent_request = clip_request(opponent_request, budget)

        game.step([own_request, opponent_request])
        agent.update(
            own_request,
            [opponent_request],
            " ".join(game.resource_tokens()),
        )

    own_final = float(Decimal(game.resources[0]) / game.scale)
    opponent_final = float(Decimal(game.resources[1]) / game.scale)
    return own_final, opponent_final, clipped_turns


def main():
    print("test strategy turns own_R opponent_R clipped_turns own_R_per_turn")
    total = 0.0
    for index in range(1, 11):
        params = parse_test_map(ROOT / "tests" / f"{index:02d}")
        own, opponent, clipped = run_one(index)
        per_turn = own / params[3]
        total += per_turn
        print(
            f"{index:02d} {params[-1]} {params[3]} "
            f"{own:.3f} {opponent:.3f} {clipped} {per_turn:.3f}"
        )
    print(f"sum_own_R_per_turn={total:.3f}")
    print("WARNING: fixed replay is diagnostic only, not an official evaluation.")


if __name__ == "__main__":
    main()
