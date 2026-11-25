"""CELL 11B.2 — signed-wind feed-forward screen on the horizontal turn channel.

This is a development screening experiment, not a final holdout.
It starts from v3b, preserves its vertical adaptive gate, and adds one
low-influence auxiliary neuron as a signed wind input to turn channel 1.

Action encoding from torch_lander/env.py:
    throttle = (action // 3) * 0.5
    turn = action % 3 - 1

The wind observation is feature 12, normalized as physical wind / 3.
The horizontal turn channel is second-layer neuron 1; its output signature
must match [-10, 0, 10] repeated for the three throttle levels.
"""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from torch_lander import Lander, sample, load, export, FAMILIES


ALPHAS = (-0.0192, -0.0096, 0.0096, 0.0192)
WIND_SHARPNESS = 2.0
TRACE_EVERY = 5


def _find_vertical_gate(model):
    w0 = model[0].weight.detach()
    mask = (
        (w0[:, 8].abs() > 50.0)
        & (w0[:, 9].abs() > 50.0)
        & ((w0[:, 8] * w0[:, 9]) < 0)
    )
    ids = torch.nonzero(mask, as_tuple=True)[0]
    if len(ids) != 1:
        raise ValueError(
            f"Expected exactly one v3b gravity/thrust gate, found {len(ids)}"
        )
    return int(ids.item())


def _make_wind_candidate(base_model, alpha, sharpness=WIND_SHARPNESS):
    model = deepcopy(base_model).cpu().eval()
    w0, b0 = model[0].weight, model[0].bias
    w2, w3 = model[2].weight, model[4].weight

    vertical_gate = _find_vertical_gate(model)

    expected_turn_signature = torch.tensor(
        [-10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0],
        dtype=w3.dtype,
    )
    if not torch.allclose(
        w3[:, 1].detach().cpu(),
        expected_turn_signature,
        atol=1e-3,
        rtol=0.0,
    ):
        raise ValueError(
            "Second-layer neuron 1 no longer matches the verified turn channel"
        )

    # Reuse the least influential auxiliary first-layer neuron, excluding
    # the two primary channels and v3b's existing vertical gate.
    influence = (w3.abs() @ w2.abs()).sum(dim=0)
    available = [
        j for j in range(2, w2.shape[1])
        if j != vertical_gate
    ]
    wind_gate = min(available, key=lambda j: float(influence[j]))

    # The new hidden neuron responds only to signed normalized wind.
    # At zero wind its output is exactly zero, preserving non-crosswind input.
    w0[wind_gate].zero_()
    b0[wind_gate].zero_()
    w0[wind_gate, 12] = float(sharpness)

    # Route signed wind to the existing horizontal turn-control neuron.
    # No bias shift: positive and negative wind produce opposite corrections.
    w2[:, wind_gate].zero_()
    w2[1, wind_gate] = float(alpha)

    return model, vertical_gate, wind_gate, float(influence[wind_gate])


def _evaluate_one(name, model_path, seed, n_episodes, horizon, trace_every):
    episodes = sample(
        n_episodes, seed, family=5, profile="medium"
    )
    env = Lander(episodes, score_mode="contest", horizon=horizon)
    model = load(model_path).eval()

    action_counts = np.zeros(9, dtype=np.int64)
    rewards = torch.zeros(n_episodes, dtype=torch.float64)
    min_abs_dx = np.full(n_episodes, np.inf, dtype=np.float64)
    min_abs_dx_low = np.full(n_episodes, np.inf, dtype=np.float64)
    trace_rows = []

    started = time.perf_counter()
    step_index = 0

    with torch.no_grad():
        while not bool(env.done.all()):
            obs = env.observe()
            actions = model(obs).argmax(dim=1)
            actions_np = actions.detach().cpu().numpy().astype(int)

            active = (~env.done).detach().cpu().numpy().astype(bool)
            state = env.state.detach().cpu().numpy().astype(float)
            params = env.params.detach().cpu().numpy().astype(float)
            steps = env.steps.detach().cpu().numpy().astype(int)

            active_ids = np.flatnonzero(active)
            for i in active_ids:
                action = int(actions_np[i])
                action_counts[action] += 1

                dx = float(state[i, 0] - params[i, 6])
                height = float(state[i, 1])
                min_abs_dx[i] = min(min_abs_dx[i], abs(dx))
                if height <= 5.0:
                    min_abs_dx_low[i] = min(min_abs_dx_low[i], abs(dx))

                if steps[i] % trace_every == 0:
                    trace_rows.append({
                        "model": name,
                        "episode_id": int(i),
                        "step": int(steps[i]),
                        "wind": float(params[i, 3]),
                        "pad_offset": dx,
                        "height": height,
                        "vx": float(state[i, 2]),
                        "vy": float(state[i, 3]),
                        "theta": float(state[i, 4]),
                        "omega": float(state[i, 5]),
                        "action": action,
                        "throttle": float((action // 3) * 0.5),
                        "turn": int(action % 3 - 1),
                    })

            _, reward, _, _, _ = env.step(actions)
            rewards += reward.detach().cpu().double()
            step_index += 1

    final_state = env.state.detach().cpu().numpy().astype(float)
    params = env.params.detach().cpu().numpy().astype(float)
    landed = env.landed.detach().cpu().numpy().astype(int)
    hull = env.hull_contact.detach().cpu().numpy().astype(int)
    timeout = env.truncated.detach().cpu().numpy().astype(int)
    terminated = env.terminated.detach().cpu().numpy().astype(bool)
    settle = env.settle_time.detach().cpu().numpy().astype(float)
    steps = env.steps.detach().cpu().numpy().astype(int)
    rewards_np = rewards.detach().cpu().numpy().astype(float)

    # Include the exact terminal state in the closest-approach metrics.
    for i in range(n_episodes):
        final_dx = float(final_state[i, 0] - params[i, 6])
        min_abs_dx[i] = min(min_abs_dx[i], abs(final_dx))
        if final_state[i, 1] <= 5.0:
            min_abs_dx_low[i] = min(min_abs_dx_low[i], abs(final_dx))

    rows = []
    for i in range(n_episodes):
        if landed[i]:
            status = "LANDED"
        elif hull[i]:
            status = "HULL_CONTACT"
        elif timeout[i]:
            status = "TIMEOUT"
        elif terminated[i] and settle[i] >= 1.0 - 1e-10:
            status = "INVALID_LANDING"
        elif terminated[i]:
            status = "OUT_OF_BOUNDS"
        else:
            status = "OTHER"

        final_dx = float(final_state[i, 0] - params[i, 6])
        rows.append({
            "model": name,
            "episode_id": i,
            "family": "crosswind",
            "return": float(rewards_np[i]),
            "landed": int(landed[i]),
            "hull_contact": int(hull[i]),
            "timeout": int(timeout[i]),
            "status": status,
            "steps": int(steps[i]),
            "wind": float(params[i, 3]),
            "pad_half_width": float(params[i, 7]),
            "final_pad_offset": final_dx,
            "final_abs_pad_offset": abs(final_dx),
            "min_abs_pad_offset": float(min_abs_dx[i]),
            "min_abs_pad_offset_below5": (
                float(min_abs_dx_low[i])
                if np.isfinite(min_abs_dx_low[i]) else np.nan
            ),
            "final_height": float(final_state[i, 1]),
            "final_vx": float(final_state[i, 2]),
            "final_vy": float(final_state[i, 3]),
            "final_theta": float(final_state[i, 4]),
            "final_omega": float(final_state[i, 5]),
            "final_settle_time": float(settle[i]),
        })

    action_rows = []
    total_actions = int(action_counts.sum())
    for action in range(9):
        action_rows.append({
            "model": name,
            "action": action,
            "throttle": float((action // 3) * 0.5),
            "turn": int(action % 3 - 1),
            "count": int(action_counts[action]),
            "percent": (
                100.0 * action_counts[action] / total_actions
                if total_actions else 0.0
            ),
        })

    df = pd.DataFrame(rows)
    elapsed = time.perf_counter() - started
    print(
        f"{name} | R={df['return'].sum():.2f} | "
        f"landed={int(df['landed'].sum())}/{n_episodes} | "
        f"hull={int(df['hull_contact'].sum())} | "
        f"invalid={(df['status'] == 'INVALID_LANDING').sum()} | "
        f"time={elapsed:.1f}s"
    )
    return df, pd.DataFrame(trace_rows), pd.DataFrame(action_rows)


def run_cell11b2(
    models_dir,
    reports_dir,
    seed=93517,
    n_episodes=48,
    horizon=800,
    n_bootstrap=5000,
    trace_every=TRACE_EVERY,
):
    if FAMILIES[5] != "crosswind":
        raise RuntimeError("Expected FAMILIES[5] to be crosswind")

    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    base_path = models_dir / "weights_v3b.pt"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing v3b model: {base_path}")

    base_model = load(base_path).eval()
    model_paths = {"v3b": base_path}
    candidate_meta = []

    for alpha in ALPHAS:
        tag = ("m" if alpha < 0 else "p") + f"{abs(alpha):.4f}".replace(".", "")
        name = f"v3b_wind_{tag}"
        candidate, vertical_gate, wind_gate, influence = _make_wind_candidate(
            base_model, alpha
        )
        candidate_path = models_dir / f"weights_{name}.pt"
        export(candidate, candidate_path)
        model_paths[name] = candidate_path
        candidate_meta.append({
            "model": name,
            "alpha": float(alpha),
            "wind_sharpness": float(WIND_SHARPNESS),
            "vertical_gate_neuron": int(vertical_gate),
            "wind_gate_neuron": int(wind_gate),
            "replaced_path_influence": float(influence),
            "base_model": "v3b",
        })

    episode_frames = []
    trace_frames = []
    action_frames = []

    for name, model_path in model_paths.items():
        df_ep, df_trace, df_actions = _evaluate_one(
            name, model_path, seed, n_episodes, horizon, trace_every
        )
        episode_frames.append(df_ep)
        trace_frames.append(df_trace)
        action_frames.append(df_actions)

    flights = pd.concat(episode_frames, ignore_index=True)
    traces = pd.concat(trace_frames, ignore_index=True)
    actions = pd.concat(action_frames, ignore_index=True)

    summary = (
        flights.groupby("model")
        .agg(
            episodes=("episode_id", "count"),
            total_return=("return", "sum"),
            landed=("landed", "sum"),
            hull_contacts=("hull_contact", "sum"),
            invalid_landings=("status", lambda x: int((x == "INVALID_LANDING").sum())),
            timeouts=("timeout", "sum"),
            mean_abs_final_pad_offset=("final_abs_pad_offset", "mean"),
            median_abs_final_pad_offset=("final_abs_pad_offset", "median"),
            mean_min_abs_pad_offset_below5=("min_abs_pad_offset_below5", "mean"),
            mean_steps=("steps", "mean"),
        )
        .reset_index()
    )

    baseline = (
        flights[flights["model"] == "v3b"]
        .set_index("episode_id")["return"]
        .sort_index()
    )
    rng = np.random.default_rng(seed + 2026)
    paired_rows = []

    for name in model_paths:
        if name == "v3b":
            continue
        candidate = (
            flights[flights["model"] == name]
            .set_index("episode_id")["return"]
            .sort_index()
        )
        delta = (candidate - baseline).to_numpy(dtype=np.float64)
        wins = int((delta > 1e-8).sum())
        losses = int((delta < -1e-8).sum())
        ties = int(len(delta) - wins - losses)
        boot = np.empty(n_bootstrap, dtype=np.float64)
        for b in range(n_bootstrap):
            boot[b] = rng.choice(delta, size=len(delta), replace=True).mean()
        low, high = np.percentile(boot, [2.5, 97.5])

        cand_rows = flights[flights["model"] == name]
        base_rows = flights[flights["model"] == "v3b"]
        paired_rows.append({
            "model": name,
            "episodes": len(delta),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "mean_delta_return": float(delta.mean()),
            "bootstrap_ci_low": float(low),
            "bootstrap_ci_high": float(high),
            "landing_delta": int(cand_rows["landed"].sum() - base_rows["landed"].sum()),
            "hull_delta": int(cand_rows["hull_contact"].sum() - base_rows["hull_contact"].sum()),
        })

    paired = pd.DataFrame(paired_rows)
    meta = pd.DataFrame(candidate_meta)

    summary.to_csv(reports_dir / "cell11b2_crosswind_summary.csv", index=False)
    flights.to_csv(reports_dir / "cell11b2_crosswind_episodes.csv", index=False)
    traces.to_csv(reports_dir / "cell11b2_trajectory_trace.csv", index=False)
    actions.to_csv(reports_dir / "cell11b2_action_distribution.csv", index=False)
    paired.to_csv(reports_dir / "cell11b2_paired_screen.csv", index=False)
    meta.to_csv(reports_dir / "cell11b2_candidate_manifest.csv", index=False)

    assert len(flights) == len(model_paths) * n_episodes
    assert len(paired) == len(ALPHAS)
    assert summary["episodes"].eq(n_episodes).all()

    print("\nCELL 11B.2 — CROSSWIND SCREEN")
    print(summary.round(3).to_string(index=False))
    print("\nPAIRED COMPARISON AGAINST v3b")
    print(paired.round(4).to_string(index=False))
    print("\nCANDIDATE MANIFEST")
    print(meta.to_string(index=False))
    print("\nQC: PASS")
    print(f"Development seed: {seed}")
    print("All 48 episodes are crosswind; do not treat this as final validation.")
    print("Original v2c and v3b checkpoints were not overwritten.")

    return summary, paired, flights, traces, actions, meta
