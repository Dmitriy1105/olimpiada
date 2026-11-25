"""CELL 11B.3 — full paired validation of v3b vs signed-wind candidate.

Compares the incumbent v3b with the preliminary CELL 11B.2 winner
v3b_wind_p00192 on a fresh balanced six-family development/validation batch.
The screening seed 93517 is not reused.

This script does not tune parameters and does not overwrite any weights.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from torch_lander import Lander, sample, load, FAMILIES


def _evaluate(name, model_path, n_episodes, seed, horizon):
    # Recreate identical scenarios for each model.
    episodes = sample(n_episodes, seed, profile="medium")
    env = Lander(episodes, score_mode="contest", horizon=horizon)
    model = load(model_path).eval()
    rewards = torch.zeros(n_episodes, dtype=torch.float64)
    started = time.perf_counter()

    with torch.no_grad():
        while not bool(env.done.all()):
            obs = env.observe()
            actions = model(obs).argmax(dim=1)
            _, reward, _, _, _ = env.step(actions)
            rewards += reward.detach().cpu().double()

    assert bool(env.done.all()), f"{name}: unfinished episodes"
    assert torch.isfinite(rewards).all(), f"{name}: non-finite rewards"

    landed = env.landed.detach().cpu().numpy().astype(int)
    hull = env.hull_contact.detach().cpu().numpy().astype(int)
    timeout = env.truncated.detach().cpu().numpy().astype(int)
    terminated = env.terminated.detach().cpu().numpy().astype(bool)
    settle = env.settle_time.detach().cpu().numpy().astype(float)
    steps = env.steps.detach().cpu().numpy().astype(int)
    state = env.state.detach().cpu().numpy().astype(float)
    params = env.params.detach().cpu().numpy().astype(float)
    family_ids = env.family.detach().cpu().numpy().astype(int)
    reward_np = rewards.detach().cpu().numpy().astype(float)

    assert state.shape == (n_episodes, 6)
    assert len(FAMILIES) == 6

    settled = settle >= 0.5 - 1e-10
    invalid = terminated & settled & (landed == 0) & (hull == 0)
    out_of_bounds = terminated & (~settled) & (landed == 0) & (hull == 0)

    rows = []
    for i in range(n_episodes):
        if landed[i]:
            status = "LANDED"
        elif hull[i]:
            status = "HULL_CONTACT"
        elif timeout[i]:
            status = "TIMEOUT"
        elif invalid[i]:
            status = "INVALID_LANDING"
        elif out_of_bounds[i]:
            status = "OUT_OF_BOUNDS"
        else:
            status = "OTHER"

        pad_offset = float(state[i, 0] - params[i, 6])
        rows.append({
            "model": name,
            "episode_id": i,
            "family": FAMILIES[int(family_ids[i])],
            "return": float(reward_np[i]),
            "landed": int(landed[i]),
            "hull_contact": int(hull[i]),
            "timeout": int(timeout[i]),
            "invalid_landing": int(invalid[i]),
            "out_of_bounds": int(out_of_bounds[i]),
            "status": status,
            "steps": int(steps[i]),
            "wind": float(params[i, 3]),
            "pad_half_width": float(params[i, 7]),
            "final_pad_offset": pad_offset,
            "final_abs_pad_offset": abs(pad_offset),
            "final_height": float(state[i, 1]),
            "final_vx": float(state[i, 2]),
            "final_vy": float(state[i, 3]),
            "final_theta": float(state[i, 4]),
            "final_omega": float(state[i, 5]),
            "final_settle_time": float(settle[i]),
        })

    df = pd.DataFrame(rows)
    elapsed = time.perf_counter() - started
    print(
        f"{name} | R={df['return'].sum():.2f} | "
        f"Landed={int(df['landed'].sum())}/{n_episodes} | "
        f"Timeout={int(df['timeout'].sum())} | "
        f"Hull={int(df['hull_contact'].sum())} | "
        f"Time={elapsed:.1f}s"
    )
    return df


def run_cell11b3(
    models_dir,
    reports_dir,
    seed=94627,
    n_episodes=240,
    horizon=800,
    n_bootstrap=20000,
):
    if n_episodes % len(FAMILIES) != 0:
        raise ValueError("n_episodes must divide evenly across all families")

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

    frames = [
        _evaluate(name, path, n_episodes, seed, horizon)
        for name, path in paths.items()
    ]
    flights = pd.concat(frames, ignore_index=True)

    summary = (
        flights.groupby("model")
        .agg(
            episodes=("episode_id", "count"),
            total_return=("return", "sum"),
            landed=("landed", "sum"),
            timeouts=("timeout", "sum"),
            hull_contacts=("hull_contact", "sum"),
            invalid_landings=("invalid_landing", "sum"),
            out_of_bounds=("out_of_bounds", "sum"),
            mean_steps=("steps", "mean"),
            mean_abs_final_pad_offset=("final_abs_pad_offset", "mean"),
        )
    )
    summary["success_pct"] = (
        100.0 * summary["landed"] / summary["episodes"]
    )

    family = (
        flights.groupby(["family", "model"])
        .agg(
            episodes=("episode_id", "count"),
            return_sum=("return", "sum"),
            landed=("landed", "sum"),
            timeouts=("timeout", "sum"),
            hull_contacts=("hull_contact", "sum"),
            invalid_landings=("invalid_landing", "sum"),
            out_of_bounds=("out_of_bounds", "sum"),
        )
        .reset_index()
    )

    paired = flights.pivot(
        index=["episode_id", "family"],
        columns="model",
        values=["return", "landed", "timeout", "hull_contact",
                "invalid_landing", "out_of_bounds"],
    )
    paired.columns = [
        f"{metric}_{model}" for metric, model in paired.columns
    ]
    paired = paired.reset_index()
    candidate = "v3b_wind_p00192"
    paired["delta_return"] = (
        paired[f"return_{candidate}"] - paired["return_v3b"]
    )
    paired["delta_landed"] = (
        paired[f"landed_{candidate}"] - paired["landed_v3b"]
    )
    paired["delta_timeout"] = (
        paired[f"timeout_{candidate}"] - paired["timeout_v3b"]
    )
    paired["delta_hull"] = (
        paired[f"hull_contact_{candidate}"] - paired["hull_contact_v3b"]
    )

    wins = int((paired["delta_return"] > 1e-8).sum())
    losses = int((paired["delta_return"] < -1e-8).sum())
    ties = int(len(paired) - wins - losses)

    # Check balanced representation and identical episode/family pairs.
    counts = paired.groupby("family").size()
    assert len(counts) == len(FAMILIES), counts.to_dict()
    assert counts.nunique() == 1, counts.to_dict()
    assert len(flights) == 2 * n_episodes
    assert len(paired) == n_episodes

    # Stratified paired bootstrap: equal weight to each family.
    rng = np.random.default_rng(seed + 20261126)
    family_deltas = [
        paired.loc[paired["family"] == fam, "delta_return"].to_numpy(
            dtype=np.float64
        )
        for fam in FAMILIES
    ]
    boot_mean = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        boot_mean[b] = np.mean([
            rng.choice(values, size=len(values), replace=True).mean()
            for values in family_deltas
        ])
    mean_ci = np.percentile(boot_mean, [2.5, 97.5])

    # Secondary bootstrap CI for the total paired reward difference.
    deltas = paired["delta_return"].to_numpy(dtype=np.float64)
    boot_total = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        boot_total[b] = rng.choice(
            deltas, size=len(deltas), replace=True
        ).sum()
    total_ci = np.percentile(boot_total, [2.5, 97.5])

    flights.to_csv(reports_dir / "cell11b3_flights.csv", index=False)
    summary.to_csv(reports_dir / "cell11b3_summary.csv")
    family.to_csv(reports_dir / "cell11b3_family.csv", index=False)
    paired.to_csv(reports_dir / "cell11b3_paired.csv", index=False)

    print("\nEPISODES BY FAMILY")
    print(counts.to_string())
    print("\nCELL 11B.3 — SUMMARY")
    print(summary.round(3).to_string())
    print("\nFAMILY PERFORMANCE")
    print(family.round(3).to_string(index=False))
    print("\nPAIRED COMPARISON: v3b_wind_p00192 minus v3b")
    print(f"Wins/losses/ties: {wins}/{losses}/{ties}")
    print(f"Mean paired delta: {deltas.mean():.4f}")
    print(f"Stratified 95% CI, mean family-weighted delta: "
          f"[{mean_ci[0]:.4f}, {mean_ci[1]:.4f}]")
    print(f"Paired 95% CI, total reward delta: "
          f"[{total_ci[0]:.3f}, {total_ci[1]:.3f}]")
    print(f"Landing delta: {int(paired['delta_landed'].sum())}")
    print(f"Hull-contact delta: {int(paired['delta_hull'].sum())}")
    print("\nTERMINAL STATUS COUNTS")
    print(
        flights.groupby(["model", "family", "status"])
        .size().unstack(fill_value=0).to_string()
    )

    print("\nQC: PASS")
    print(f"Seed: {seed} | episodes/model: {n_episodes} | horizon: {horizon}")
    print("Seed 93517 from CELL 11B.2 was not reused.")
    print("No model weights were overwritten.")
    print("Promotion decision must consider family-level regressions, not only total reward.")

    return flights, summary, family, paired
