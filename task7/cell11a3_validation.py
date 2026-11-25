"""CELL 11A.3 — paired validation of v2c and v3b.

Paste the cell body into the participant notebook, or import this module
and call run_cell11a_validation(MODELS_DIR, REPORTS_DIR).
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from torch_lander import Lander, sample, load, FAMILIES


def _evaluate(name, weights_path, n_episodes, seed, horizon):
    model = load(weights_path).eval()
    env = Lander(
        sample(n_episodes, seed, profile="medium"),
        score_mode="contest",
        horizon=horizon,
    )
    rewards = torch.zeros(n_episodes, dtype=torch.float64)
    started = time.perf_counter()

    with torch.no_grad():
        while not bool(env.done.all()):
            obs = env.observe()
            actions = model(obs).argmax(dim=1)
            _, reward, _, _, _ = env.step(actions)
            rewards += reward.double()

    assert bool(env.done.all())
    assert torch.isfinite(rewards).all()

    rows = []
    for i in range(n_episodes):
        rows.append({
            "model": name,
            "episode_id": i,
            "family": FAMILIES[int(env.family[i])],
            "return": float(rewards[i]),
            "landed": int(env.landed[i]),
            "timeout": int(env.truncated[i]),
            "hull_contact": int(env.hull_contact[i]),
            "steps": int(env.steps[i]),
        })

    print(
        f"{name} | R={float(rewards.sum()):.2f} | "
        f"Landed={int(env.landed.sum())}/{n_episodes} | "
        f"Timeout={int(env.truncated.sum())} | "
        f"Hull={int(env.hull_contact.sum())} | "
        f"Time={time.perf_counter()-started:.1f}s"
    )
    return pd.DataFrame(rows)


def run_cell11a_validation(
    models_dir,
    reports_dir,
    seed=81739,
    n_episodes=60,
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
    for p in paths.values():
        assert p.exists(), f"Missing model: {p}"

    # Each model receives the same deterministic sampled episode batch.
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
            mean_steps=("steps", "mean"),
        )
    )
    summary["success_pct"] = 100 * summary["landed"] / summary["episodes"]

    family = (
        flights.groupby(["family", "model"])
        .agg(
            episodes=("episode_id", "count"),
            return_sum=("return", "sum"),
            landed=("landed", "sum"),
            timeouts=("timeout", "sum"),
            hull_contacts=("hull_contact", "sum"),
        )
        .reset_index()
    )

    paired = flights.pivot(
        index=["episode_id", "family"],
        columns="model",
        values=["return", "landed", "timeout", "hull_contact"],
    )
    paired.columns = [f"{metric}_{model}" for metric, model in paired.columns]
    paired = paired.reset_index()
    paired["delta_return"] = paired["return_v3b"] - paired["return_v2c"]
    paired["delta_landed"] = paired["landed_v3b"] - paired["landed_v2c"]

    wins = int((paired["delta_return"] > 1e-8).sum())
    losses = int((paired["delta_return"] < -1e-8).sum())
    ties = n_episodes - wins - losses

    # Stratified paired bootstrap: resample within each flight family,
    # preserving the intended equal weight of all six families.
    rng = np.random.default_rng(20261125)
    family_deltas = [
        paired.loc[paired["family"] == fam, "delta_return"].to_numpy()
        for fam in FAMILIES
    ]
    assert all(len(x) > 0 for x in family_deltas), "Missing family in sample"

    bootstrap = np.empty(10000, dtype=np.float64)
    for b in range(len(bootstrap)):
        family_means = [
            rng.choice(x, size=len(x), replace=True).mean()
            for x in family_deltas
        ]
        bootstrap[b] = np.mean(family_means)

    ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])

    flights.to_csv(reports_dir / "cell11a3_flights.csv", index=False)
    summary.to_csv(reports_dir / "cell11a3_summary.csv")
    family.to_csv(reports_dir / "cell11a3_family.csv", index=False)
    paired.to_csv(reports_dir / "cell11a3_paired.csv", index=False)

    assert len(flights) == 2 * n_episodes
    assert len(paired) == n_episodes
    assert wins + losses + ties == n_episodes
    assert summary.loc["v2c", "episodes"] == n_episodes
    assert summary.loc["v3b", "episodes"] == n_episodes

    print("\nCELL 11A.3 — SUMMARY")
    print(summary.round(3))
    print(f"Paired wins/losses/ties: {wins}/{losses}/{ties}")
    print(f"Mean paired delta per flight: {paired['delta_return'].mean():.3f}")
    print(f"Stratified bootstrap 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print("QC: PASS")

    r2 = float(summary.loc["v2c", "total_return"])
    r3 = float(summary.loc["v3b", "total_return"])
    if r3 > r2 and ci_low > 0:
        print("EXPERIMENT: IMPROVEMENT SUPPORTED")
    elif r3 > r2:
        print("EXPERIMENT: IMPROVED, BUT UNCERTAIN")
    else:
        print("EXPERIMENT: NOT IMPROVED")

    return flights, summary, family, paired
