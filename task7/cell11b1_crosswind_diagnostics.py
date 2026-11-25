"""CELL 11B.1 — Crosswind and recovery diagnostics for v2c/v3b.

Run in the participant notebook:
    from cell11b1_crosswind_diagnostics import run_cell11b1
    df_episodes, df_actions, df_family = run_cell11b1(MODELS_DIR, REPORTS_DIR)

This is a development diagnostic seed, not a final holdout.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from torch_lander import Lander, sample, load, FAMILIES


FEATURE_NAMES = [
    "relative_x", "height", "vx", "vy", "sin_theta", "cos_theta",
    "omega", "remaining_time", "gravity", "main_thrust",
    "left_turn_power", "right_turn_power", "wind", "linear_drag",
    "angular_drag", "pad_half_width", "absolute_x",
] + [f"terrain_height_{i:02d}" for i in range(18)] + [
    "left_leg_angle", "right_leg_angle",
    "left_leg_velocity", "right_leg_velocity",
    "left_leg_contact", "right_leg_contact", "settle_time",
]

assert len(FEATURE_NAMES) == 42


def _as_cpu_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.as_tensor(x)


def _evaluate_with_trace(name, model_path, n_episodes, seed, horizon):
    episodes = sample(n_episodes, seed, profile="medium")
    env = Lander(episodes, score_mode="contest", horizon=horizon)
    model = load(model_path).eval()

    family_ids = _as_cpu_tensor(env.family).long().numpy()
    family_names = [FAMILIES[int(i)] for i in family_ids]

    initial_obs = env.observe().detach().cpu().numpy().copy()
    action_counts = torch.zeros((n_episodes, 9), dtype=torch.int64)
    rewards = torch.zeros(n_episodes, dtype=torch.float64)

    started = time.perf_counter()
    step = 0

    with torch.no_grad():
        while not bool(env.done.all()):
            obs = env.observe()
            active = (~env.done).detach().cpu()
            actions = model(obs).argmax(dim=1)
            actions_cpu = actions.detach().cpu().long()

            active_ids = torch.nonzero(active, as_tuple=True)[0]
            if len(active_ids):
                encoded = active_ids * 9 + actions_cpu[active_ids]
                action_counts += torch.bincount(
                    encoded, minlength=n_episodes * 9
                ).reshape(n_episodes, 9)

            _, reward, _, _, _ = env.step(actions)
            rewards += reward.detach().cpu().double()
            step += 1

            if step % 200 == 0:
                print(
                    f"{name} | step={step} | "
                    f"finished={int(env.done.sum())}/{n_episodes} | "
                    f"landed={int(env.landed.sum())}"
                )

    final_obs = env.observe().detach().cpu().numpy().copy()
    assert initial_obs.shape == (n_episodes, 42)
    assert final_obs.shape == (n_episodes, 42)
    assert np.isfinite(initial_obs).all() and np.isfinite(final_obs).all()

    landed = _as_cpu_tensor(env.landed).numpy().astype(int)
    timeout = _as_cpu_tensor(env.truncated).numpy().astype(int)
    hull = _as_cpu_tensor(env.hull_contact).numpy().astype(int)
    steps = _as_cpu_tensor(env.steps).numpy().astype(int)
    # The participant-v2 Lander exposes physical state as env.state, not
    # separate env.x/env.y attributes. state columns are x, nominal height,
    # vx, vy, theta, omega; settle_time/terminated are documented dynamic fields.
    final_state = _as_cpu_tensor(env.state).numpy().astype(float)
    final_x = final_state[:, 0]
    final_y = final_state[:, 1]
    final_settle = _as_cpu_tensor(env.settle_time).numpy().astype(float)
    terminated = _as_cpu_tensor(env.terminated).numpy().astype(bool)

    assert final_state.shape == (n_episodes, 6)
    assert final_settle.shape == (n_episodes,)
    assert terminated.shape == (n_episodes,)

    # Reconstruct terminal reason from the exact termination logic in
    # participant-v2 torch_lander/env.py: terminated = crash | outside | settled.
    # hull_contact identifies crash; settle_time identifies settled.
    settled = final_settle >= 0.5 - 1e-10
    landed_mask = landed.astype(bool)
    hull_mask = hull.astype(bool)
    invalid_landing = terminated & settled & ~landed_mask & ~hull_mask
    out_of_bounds = terminated & ~settled & ~hull_mask
    status = np.full(n_episodes, "OTHER", dtype=object)
    status[invalid_landing] = "INVALID_LANDING"
    status[out_of_bounds] = "OUT_OF_BOUNDS"
    status[timeout == 1] = "TIMEOUT"
    status[hull == 1] = "HULL_CONTACT"
    status[landed == 1] = "LANDED"

    rows = []
    for i in range(n_episodes):
        row = {
            "model": name,
            "episode_id": i,
            "family": family_names[i],
            "return": float(rewards[i]),
            "landed": int(landed[i]),
            "timeout": int(timeout[i]),
            "hull_contact": int(hull[i]),
            "steps": int(steps[i]),
            "status": str(status[i]),
            "final_x": float(final_x[i]),
            "final_y": float(final_y[i]),
            "final_settle_time": float(final_settle[i]),
        }
        for j, feature in enumerate(FEATURE_NAMES):
            row[f"initial_{feature}"] = float(initial_obs[i, j])
            row[f"final_{feature}"] = float(final_obs[i, j])
        for a in range(9):
            row[f"action_{a}_count"] = int(action_counts[i, a])
        rows.append(row)

    action_rows = []
    for family in FAMILIES:
        mask = np.asarray(family_names) == family
        counts = action_counts.numpy()[mask].sum(axis=0)
        total = int(counts.sum())
        for action in range(9):
            action_rows.append({
                "model": name,
                "family": family,
                "action": action,
                "throttle": (action // 3) * 0.5,
                "turn": action % 3 - 1,
                "count": int(counts[action]),
                "percent": 100.0 * counts[action] / total if total else 0.0,
            })

    elapsed = time.perf_counter() - started
    print(
        f"{name} | R={float(rewards.sum()):.2f} | "
        f"landed={int(landed.sum())}/{n_episodes} | "
        f"timeout={int(timeout.sum())} | hull={int(hull.sum())} | "
        f"time={elapsed:.1f}s"
    )
    return pd.DataFrame(rows), pd.DataFrame(action_rows)


def run_cell11b1(
    models_dir,
    reports_dir,
    seed=73129,
    n_episodes=240,
    horizon=800,
):
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    paths = {
        "v2c": models_dir / "weights_v2c.pt",
        "v3b": models_dir / "weights_v3b.pt",
    }
    for path in paths.values():
        assert path.exists(), f"Missing model: {path}"

    # Same deterministic scenario batch is recreated for each model.
    episode_frames = []
    action_frames = []
    for name, path in paths.items():
        df_ep, df_act = _evaluate_with_trace(
            name, path, n_episodes, seed, horizon
        )
        episode_frames.append(df_ep)
        action_frames.append(df_act)

    episodes = pd.concat(episode_frames, ignore_index=True)
    actions = pd.concat(action_frames, ignore_index=True)

    family_summary = (
        episodes.groupby(["model", "family"])
        .agg(
            episodes=("episode_id", "count"),
            total_return=("return", "sum"),
            landed=("landed", "sum"),
            timeouts=("timeout", "sum"),
            hull_contacts=("hull_contact", "sum"),
            mean_steps=("steps", "mean"),
            mean_initial_wind=("initial_wind", "mean"),
            mean_initial_vx=("initial_vx", "mean"),
            mean_initial_vy=("initial_vy", "mean"),
            mean_final_relative_x=("final_relative_x", "mean"),
            mean_final_height=("final_height", "mean"),
            mean_final_vx=("final_vx", "mean"),
            mean_final_vy=("final_vy", "mean"),
        )
        .reset_index()
    )

    # Print detailed failure records for the two target families.
    targets = episodes[
        episodes["family"].isin(["crosswind", "recovery"])
    ].copy()

    episodes.to_csv(reports_dir / "cell11b1_episode_diagnostics.csv", index=False)
    actions.to_csv(reports_dir / "cell11b1_action_distribution.csv", index=False)
    family_summary.to_csv(reports_dir / "cell11b1_family_summary.csv", index=False)
    targets.to_csv(reports_dir / "cell11b1_crosswind_recovery_detail.csv", index=False)

    assert len(episodes) == 2 * n_episodes
    assert len(actions) == 2 * len(FAMILIES) * 9
    assert family_summary.groupby("model")["episodes"].sum().eq(n_episodes).all()

    print("\nFAMILY SUMMARY")
    print(family_summary.round(3).to_string(index=False))

    print("\nTERMINAL STATUS COUNTS BY FAMILY")
    status_summary = (
        episodes.groupby(["model", "family", "status"])
        .size().unstack(fill_value=0).reset_index()
    )
    print(status_summary.to_string(index=False))

    print("\nCROSSWIND / RECOVERY FAILURE DETAIL")
    cols = [
        "model", "episode_id", "family", "status", "return",
        "initial_wind", "initial_relative_x", "initial_vx",
        "initial_vy", "final_relative_x", "final_height",
        "final_vx", "final_vy", "final_sin_theta",
        "final_omega", "final_x", "final_y", "final_settle_time", "steps",
    ]
    print(targets[cols].round(3).to_string(index=False))

    print("\nACTION DISTRIBUTION — CROSSWIND")
    print(
        actions[actions["family"] == "crosswind"]
        .pivot(index=["model", "action"], columns="family", values="percent")
        .round(3)
        .to_string()
    )

    print("\nNOTE: initial_*/final_* feature columns are normalized model observations.")
    print("Terminal statuses use the same ordered rules as CELL 09.")
    print("\nQC: PASS")
    print("Development seed:", seed)
    print("Final holdout: DO NOT use this seed for final claims.")
    return episodes, actions, family_summary
