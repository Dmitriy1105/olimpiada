"""CELL 11B.4 — terminal-state and trajectory diagnostics.

Compare v3b and v3b_wind_p00192 on fresh development episodes from
crosswind and low_thrust. This is diagnostic only: it does not tune or
promote a model and does not overwrite model weights.

Outputs:
  cell11b4_flights.csv
  cell11b4_trajectory_trace.csv
  cell11b4_action_distribution.csv
  cell11b4_family_summary.csv
  cell11b4_terminal_status.csv
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from torch_lander import Lander, sample, load, FAMILIES


FAMILY_NAMES = ("crosswind", "low_thrust")
TRACE_EVERY = 5
TRACE_HEIGHT = 8.0


def _status(landed, hull, timeout, terminated, settle):
    if landed:
        return "LANDED"
    if hull:
        return "HULL_CONTACT"
    if timeout:
        return "TIMEOUT"
    if terminated and settle >= 0.5 - 1e-10:
        return "INVALID_LANDING"
    if terminated:
        return "OUT_OF_BOUNDS"
    return "OTHER"


def _evaluate_model(name, model_path, seed, n_episodes, horizon, trace_every):
    model = load(model_path).eval()
    flight_rows = []
    trace_rows = []
    action_rows = []

    for family_name in FAMILY_NAMES:
        if family_name not in FAMILIES:
            raise ValueError(
                f"Family {family_name!r} not found in FAMILIES={FAMILIES}"
            )
        family_id = FAMILIES.index(family_name)
        episodes = sample(
            n_episodes, seed, family=family_id, profile="medium"
        )
        env = Lander(episodes, score_mode="contest", horizon=horizon)

        rewards = torch.zeros(n_episodes, dtype=torch.float64)
        action_counts = np.zeros(9, dtype=np.int64)
        min_abs_dx = np.full(n_episodes, np.inf, dtype=np.float64)
        min_abs_dx_low = np.full(n_episodes, np.inf, dtype=np.float64)
        started = time.perf_counter()

        with torch.no_grad():
            while not bool(env.done.all()):
                obs = env.observe()
                actions = model(obs).argmax(dim=1)
                actions_np = actions.detach().cpu().numpy().astype(int)

                active = (~env.done).detach().cpu().numpy().astype(bool)
                pre_state = env.state.detach().cpu().numpy().astype(float)
                params = env.params.detach().cpu().numpy().astype(float)
                pre_steps = env.steps.detach().cpu().numpy().astype(int)

                _, step_reward, _, _, _ = env.step(actions)

                post_state = env.state.detach().cpu().numpy().astype(float)
                post_steps = env.steps.detach().cpu().numpy().astype(int)
                post_done = env.done.detach().cpu().numpy().astype(bool)
                reward_np = step_reward.detach().cpu().numpy().astype(float)
                rewards += step_reward.detach().cpu().double()

                for i in np.flatnonzero(active):
                    action = int(actions_np[i])
                    action_counts[action] += 1

                    pad_center = float(params[i, 6])
                    wind = float(params[i, 3])
                    half_width = float(params[i, 7])
                    pre_dx = float(pre_state[i, 0] - pad_center)
                    post_dx = float(post_state[i, 0] - pad_center)
                    pre_height = float(pre_state[i, 1])
                    post_height = float(post_state[i, 1])

                    min_abs_dx[i] = min(
                        min_abs_dx[i], abs(pre_dx), abs(post_dx)
                    )
                    if pre_height <= TRACE_HEIGHT:
                        min_abs_dx_low[i] = min(
                            min_abs_dx_low[i], abs(pre_dx)
                        )
                    if post_height <= TRACE_HEIGHT:
                        min_abs_dx_low[i] = min(
                            min_abs_dx_low[i], abs(post_dx)
                        )

                    keep_trace = (
                        pre_steps[i] % trace_every == 0
                        or pre_height <= TRACE_HEIGHT
                        or post_height <= TRACE_HEIGHT
                        or post_done[i]
                    )
                    if keep_trace:
                        trace_rows.append({
                            "model": name,
                            "family": family_name,
                            "episode_id": int(i),
                            "step": int(pre_steps[i]),
                            "next_step": int(post_steps[i]),
                            "wind": wind,
                            "pad_center": pad_center,
                            "pad_half_width": half_width,
                            "action": action,
                            "throttle": float((action // 3) * 0.5),
                            "turn": int(action % 3 - 1),
                            "step_reward": float(reward_np[i]),
                            "pre_pad_offset": pre_dx,
                            "post_pad_offset": post_dx,
                            "pre_x": float(pre_state[i, 0]),
                            "pre_y": pre_height,
                            "pre_vx": float(pre_state[i, 2]),
                            "pre_vy": float(pre_state[i, 3]),
                            "pre_theta": float(pre_state[i, 4]),
                            "pre_omega": float(pre_state[i, 5]),
                            "post_x": float(post_state[i, 0]),
                            "post_y": post_height,
                            "post_vx": float(post_state[i, 2]),
                            "post_vy": float(post_state[i, 3]),
                            "post_theta": float(post_state[i, 4]),
                            "post_omega": float(post_state[i, 5]),
                            "terminal_after_step": bool(post_done[i]),
                        })

        assert bool(env.done.all()), f"{name}/{family_name}: unfinished episodes"

        final_state = env.state.detach().cpu().numpy().astype(float)
        params = env.params.detach().cpu().numpy().astype(float)
        landed = env.landed.detach().cpu().numpy().astype(int)
        hull = env.hull_contact.detach().cpu().numpy().astype(int)
        timeout = env.truncated.detach().cpu().numpy().astype(int)
        terminated = env.terminated.detach().cpu().numpy().astype(bool)
        settle = env.settle_time.detach().cpu().numpy().astype(float)
        steps = env.steps.detach().cpu().numpy().astype(int)
        rewards_np = rewards.detach().cpu().numpy().astype(float)

        for i in range(n_episodes):
            final_dx = float(final_state[i, 0] - params[i, 6])
            min_abs_dx[i] = min(min_abs_dx[i], abs(final_dx))
            if final_state[i, 1] <= TRACE_HEIGHT:
                min_abs_dx_low[i] = min(
                    min_abs_dx_low[i], abs(final_dx)
                )

            status = _status(
                bool(landed[i]), bool(hull[i]), bool(timeout[i]),
                bool(terminated[i]), float(settle[i])
            )
            flight_rows.append({
                "model": name,
                "family": family_name,
                "episode_id": int(i),
                "return": float(rewards_np[i]),
                "status": status,
                "landed": int(landed[i]),
                "hull_contact": int(hull[i]),
                "timeout": int(timeout[i]),
                "invalid_landing": int(status == "INVALID_LANDING"),
                "out_of_bounds": int(status == "OUT_OF_BOUNDS"),
                "steps": int(steps[i]),
                "wind": float(params[i, 3]),
                "pad_half_width": float(params[i, 7]),
                "final_pad_offset": final_dx,
                "final_abs_pad_offset": abs(final_dx),
                "min_abs_pad_offset": float(min_abs_dx[i]),
                "min_abs_pad_offset_below8": (
                    float(min_abs_dx_low[i])
                    if np.isfinite(min_abs_dx_low[i]) else np.nan
                ),
                "final_x": float(final_state[i, 0]),
                "final_y": float(final_state[i, 1]),
                "final_vx": float(final_state[i, 2]),
                "final_vy": float(final_state[i, 3]),
                "final_theta": float(final_state[i, 4]),
                "final_omega": float(final_state[i, 5]),
                "final_settle_time": float(settle[i]),
            })

        total_actions = int(action_counts.sum())
        for action in range(9):
            action_rows.append({
                "model": name,
                "family": family_name,
                "action": action,
                "throttle": float((action // 3) * 0.5),
                "turn": int(action % 3 - 1),
                "count": int(action_counts[action]),
                "percent": (
                    100.0 * action_counts[action] / total_actions
                    if total_actions else 0.0
                ),
            })

        family_flights = [
            row for row in flight_rows
            if row["model"] == name and row["family"] == family_name
        ]
        print(
            f"{name} | {family_name} | "
            f"R={sum(x['return'] for x in family_flights):.2f} | "
            f"landed={sum(x['landed'] for x in family_flights)}/{n_episodes} | "
            f"hull={sum(x['hull_contact'] for x in family_flights)} | "
            f"time={time.perf_counter()-started:.1f}s"
        )

    return (
        pd.DataFrame(flight_rows),
        pd.DataFrame(trace_rows),
        pd.DataFrame(action_rows),
    )


def run_cell11b4(
    models_dir,
    reports_dir,
    seed=271828,
    n_episodes=40,
    horizon=800,
    trace_every=TRACE_EVERY,
):
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    paths = {
        "v3b": models_dir / "weights_v3b.pt",
        "v3b_wind_p00192": models_dir / "weights_v3b_wind_p00192.pt",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {path}")

    all_flights = []
    all_traces = []
    all_actions = []
    for name, path in paths.items():
        flights, traces, actions = _evaluate_model(
            name, path, seed, n_episodes, horizon, trace_every
        )
        all_flights.append(flights)
        all_traces.append(traces)
        all_actions.append(actions)

    flights = pd.concat(all_flights, ignore_index=True)
    traces = pd.concat(all_traces, ignore_index=True)
    actions = pd.concat(all_actions, ignore_index=True)

    expected = len(paths) * len(FAMILY_NAMES) * n_episodes
    assert len(flights) == expected, (len(flights), expected)
    assert not flights.duplicated(
        ["model", "family", "episode_id"]
    ).any()
    assert set(flights["family"]) == set(FAMILY_NAMES)
    assert np.isfinite(
        flights[["return", "final_x", "final_y", "final_vx",
                 "final_vy", "final_theta", "final_omega"]].to_numpy()
    ).all()

    family_summary = (
        flights.groupby(["family", "model"])
        .agg(
            episodes=("episode_id", "count"),
            return_sum=("return", "sum"),
            landed=("landed", "sum"),
            hull_contacts=("hull_contact", "sum"),
            invalid_landings=("invalid_landing", "sum"),
            mean_abs_final_offset=("final_abs_pad_offset", "mean"),
            median_abs_final_offset=("final_abs_pad_offset", "median"),
            mean_min_abs_offset_below8=("min_abs_pad_offset_below8", "mean"),
            mean_abs_final_vx=("final_vx", lambda x: x.abs().mean()),
            mean_abs_final_vy=("final_vy", lambda x: x.abs().mean()),
            mean_abs_final_theta=("final_theta", lambda x: x.abs().mean()),
            mean_abs_final_omega=("final_omega", lambda x: x.abs().mean()),
        )
        .reset_index()
    )
    terminal = (
        flights.groupby(["model", "family", "status"])
        .size().unstack(fill_value=0).reset_index()
    )

    flights.to_csv(reports_dir / "cell11b4_flights.csv", index=False)
    traces.to_csv(reports_dir / "cell11b4_trajectory_trace.csv", index=False)
    actions.to_csv(reports_dir / "cell11b4_action_distribution.csv", index=False)
    family_summary.to_csv(
        reports_dir / "cell11b4_family_summary.csv", index=False
    )
    terminal.to_csv(reports_dir / "cell11b4_terminal_status.csv", index=False)

    print("\nCELL 11B.4 — FAMILY SUMMARY")
    print(family_summary.round(4).to_string(index=False))
    print("\nTERMINAL STATUS")
    print(terminal.to_string(index=False))
    print("\nACTION DISTRIBUTION")
    print(actions.round(3).to_string(index=False))
    print("\nQC: PASS")
    print(
        f"Seed={seed} | episodes/family/model={n_episodes} | "
        f"families={','.join(FAMILY_NAMES)} | horizon={horizon}"
    )
    print("Diagnostic only: no tuning, promotion, or weight overwrite.")
    print("This seed is not a final holdout.")

    return flights, traces, actions, family_summary, terminal
