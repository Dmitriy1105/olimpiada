"""Compare v1 and v4 on the same fixed-history diagnostic replay.

This is not an official score: the opponent's historical requests do not adapt
to our changed actions, and requests are clipped when the replayed opponent
cannot afford the recorded amount.
"""
from decimal import Decimal
from pathlib import Path

try:
    from .simulate import Game, parse_test_map, parse_request
    from .solution import MarsAgent as V4Agent
    from .solution_v1 import MarsAgent as V1Agent
except ImportError:
    from simulate import Game, parse_test_map, parse_request
    from solution import MarsAgent as V4Agent
    from solution_v1 import MarsAgent as V1Agent

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


def run_one(index, agent_class):
    params = parse_test_map(ROOT / "tests" / f"{index:02d}")
    game = Game(params)
    agent = agent_class(make_agent_initial(params))
    history = (ROOT / "histories" / f"{index:02d}.txt").read_text(
        encoding="ascii"
    ).splitlines()

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

    return (
        float(Decimal(game.resources[0]) / game.scale),
        clipped_turns,
        game.turns,
        params[-1],
    )


def main():
    totals = {"v1": 0.0, "v4": 0.0}
    print("map strategy turns v1_R/T v4_R/T v1_clip v4_clip")
    for index in range(1, 11):
        old = run_one(index, V1Agent)
        new = run_one(index, V4Agent)
        old_rate = old[0] / old[2]
        new_rate = new[0] / new[2]
        totals["v1"] += old_rate
        totals["v4"] += new_rate
        print(
            f"{index:02d} {old[3]} {old[2]} "
            f"{old_rate:.3f} {new_rate:.3f} {old[1]} {new[1]}"
        )
    print(f"sum_v1_R_per_turn={totals['v1']:.3f}")
    print(f"sum_v4_R_per_turn={totals['v4']:.3f}")
    if totals["v1"]:
        print(f"relative_change_pct={(totals['v4'] / totals['v1'] - 1) * 100:.3f}")
    print("WARNING: diagnostic fixed-history replay only; not an official score.")


if __name__ == "__main__":
    main()
