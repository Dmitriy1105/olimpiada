"""CELL 11A.4 — expanded paired validation of v2c and v3b.

Usage in the participant notebook:
    from cell11a4_expanded_validation import run_cell11a4
    df_flights, df_summary, df_family, df_paired = run_cell11a4(
        MODELS_DIR, REPORTS_DIR
    )
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from torch_lander import Lander, sample, load, FAMILIES


def _run_one(name, model_path, n_episodes, seed, horizon):
    # Recreate the same seeded scenarios for each model.
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


def run_cell11a4(
    models_dir,
    reports_dir,
    seed=62814,
    n_episodes=240,
    horizon=800,
    n_bootstrap=20000,
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
        assert path.exists(), f"Missing weights: {path}"

    frames = [
        _run_one(name, path, n_episodes, seed, horizon)
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
    paired.columns = [
        f"{metric}_{model}" for metric, model in paired.columns
    ]
    paired = paired.reset_index()
    paired["delta_return"] = paired["return_v3b"] - paired["return_v2c"]
    paired["delta_landed"] = paired["landed_v3b"] - paired["landed_v2c"]
    paired["delta_timeout"] = paired["timeout_v3b"] - paired["timeout_v2c"]
    paired["delta_hull"] = paired["hull_contact_v3b"] - paired["hull_contact_v2c"]

    wins = int((paired["delta_return"] > 1e-8).sum())
    losses = int((paired["delta_return"] < -1e-8).sum())
    ties = n_episodes - wins - losses

    # Verify that the seeded batch has balanced family representation.
    counts = paired.groupby("family").size()
    print("\nEpisodes by family:")
    print(counts)
    assert len(counts) == len(FAMILIES)
    assert counts.nunique() == 1, "Expected balanced family counts"

    # Stratified paired bootstrap: sample episode deltas within each
    # family, then average family means with equal family weights.
    rng = np.random.default_rng(20261126)
    family_deltas = [
        paired.loc[paired["family"] == fam, "delta_return"].to_numpy()
        for fam in FAMILIES
    ]
    bootstrap = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        means = [
            rng.choice(values, size=len(values), replace=True).mean()
            for values in family_deltas
        ]
        bootstrap[b] = np.mean(means)

    ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])

    # A secondary paired bootstrap CI for the total reward delta.
    total_bootstrap = np.empty(n_bootstrap, dtype=np.float64)
    deltas = paired["delta_return"].to_numpy()
    for b in range(n_bootstrap):
        total_bootstrap[b] = rng.choice(
            deltas, size=n_episodes, replace=True
        ).sum()
    total_ci = np.percentile(total_bootstrap, [2.5, 97.5])

    flights.to_csv(reports_dir / "cell11a4_flights.csv", index=False)
    summary.to_csv(reports_dir / "cell11a4_summary.csv")
    family.to_csv(reports_dir / "cell11a4_family.csv", index=False)
    paired.to_csv(reports_dir / "cell11a4_paired.csv", index=False)

    assert len(flights) == 2 * n_episodes
    assert len(paired) == n_episodes
    assert wins + losses + ties == n_episodes

    r2 = float(summary.loc["v2c", "total_return"])
    r3 = float(summary.loc["v3b", "total_return"])
    l2 = int(summary.loc["v2c", "landed"])
    l3 = int(summary.loc["v3b", "landed"])

    print("\nCELL 11A.4 — SUMMARY")
    print(summary.round(3))
    print(f"Paired wins/losses/ties: {wins}/{losses}/{ties}")
    print(f"Mean paired delta per flight: {deltas.mean():.3f}")
    print(f"Stratified 95% CI (mean/family): [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"Paired 95% CI (total reward delta): [{total_ci[0]:.3f}, {total_ci[1]:.3f}]")
    print(f"Landing delta: {l3-l2:+d}")
    print("QC: PASS")

    if r3 > r2 and ci_low > 0 and l3 >= l2:
        print("EXPERIMENT: IMPROVEMENT SUPPORTED")
    elif r3 > r2:
        print("EXPERIMENT: POSITIVE TREND, UNCERTAIN")
    else:
        print("EXPERIMENT: NOT IMPROVED")

    return flights, summary, family, paired
