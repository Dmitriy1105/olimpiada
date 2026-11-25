"""CELL 11B.11 — package the validated p00384 checkpoint.

Copies the exact checkpoint used in CELL 11B.10 into a separate submission
folder as weights.pt, verifies its SHA-256, reloads it with torch_lander.load,
checks the expected 42 -> 32 -> 32 -> 9 architecture and performs a
four-observation inference smoke test (no simulator steps).

The source checkpoint and any existing weights.pt are never overwritten.
"""
from pathlib import Path
import hashlib
import json
import shutil

import torch
from torch_lander import Lander, sample, load


MODEL_NAME = "v3b_wind_p00384"
SOURCE_FILENAME = "weights_v3b_wind_p00384.pt"
EXPECTED_SHA256 = (
    "ef5a6b8d9b2ba95712a3b0dc02551dd8ad4aec8cdd43d1cb4bcd865d17195c23"
)
VALIDATION_SEED = 681204
VALIDATION_EPISODES = 240
VALIDATION_HORIZON = 800

EXPECTED_SHAPES = {
    "0.weight": (32, 42),
    "0.bias": (32,),
    "2.weight": (32, 32),
    "2.bias": (32,),
    "4.weight": (9, 32),
    "4.bias": (9,),
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable(path, content):
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"Refusing to overwrite existing non-identical file: {path}"
            )
        return
    path.write_text(content, encoding="utf-8")


def run_cell11b11(models_dir, output_dir):
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    source = models_dir / SOURCE_FILENAME
    target = output_dir / "weights.pt"

    if not source.is_file():
        raise FileNotFoundError(f"Validated source checkpoint not found: {source}")

    source_hash = _sha256(source)
    if source_hash != EXPECTED_SHA256:
        raise RuntimeError(
            "Checkpoint hash mismatch. This is not the exact p00384 file "
            "validated in CELL 11B.10.\n"
            f"Expected: {EXPECTED_SHA256}\nActual:   {source_hash}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    if target.exists():
        target_hash = _sha256(target)
        if target_hash != EXPECTED_SHA256:
            raise FileExistsError(
                f"Refusing to overwrite a different weights.pt: {target}\n"
                f"Existing SHA-256: {target_hash}"
            )
    else:
        shutil.copy2(source, target)

    packaged_hash = _sha256(target)
    if packaged_hash != EXPECTED_SHA256:
        raise RuntimeError("Packaged checkpoint hash does not match validation")

    # Validate through the same participant-package loader.
    model = load(target).eval()
    state = model.state_dict()

    if set(state) != set(EXPECTED_SHAPES):
        raise AssertionError(
            f"Unexpected state_dict keys: {sorted(state)}"
        )

    for key, expected_shape in EXPECTED_SHAPES.items():
        tensor = state[key]
        if tuple(tensor.shape) != expected_shape:
            raise AssertionError(
                f"{key}: expected shape {expected_shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.float32:
            raise AssertionError(f"{key}: expected float32, got {tensor.dtype}")
        if not bool(torch.isfinite(tensor).all()):
            raise AssertionError(f"{key}: non-finite tensor values")

    # Inference-only smoke test: no environment step or weight update.
    smoke_episodes = sample(4, seed=681205, profile="medium")
    smoke_env = Lander(smoke_episodes, score_mode="contest", horizon=800)
    observation = smoke_env.observe()

    if tuple(observation.shape) != (4, 42):
        raise AssertionError(
            f"Unexpected observation shape: {tuple(observation.shape)}"
        )

    with torch.inference_mode():
        logits = model(observation)
        actions = logits.argmax(dim=1)

    if tuple(logits.shape) != (4, 9):
        raise AssertionError(f"Unexpected output shape: {tuple(logits.shape)}")
    if not bool(torch.isfinite(logits).all()):
        raise AssertionError("Non-finite inference output")
    if int(actions.min()) < 0 or int(actions.max()) > 8:
        raise AssertionError("Action outside the allowed range 0..8")

    manifest = {
        "artifact": "weights.pt",
        "model": MODEL_NAME,
        "base_model": "v3b",
        "wind_alpha": 0.0384,
        "sha256": packaged_hash,
        "architecture": "42 -> 32 -> 32 -> 9",
        "dtype": "float32",
        "validation": {
            "seed": VALIDATION_SEED,
            "episodes_per_model": VALIDATION_EPISODES,
            "episodes_per_family": 40,
            "families": [
                "approach", "crosswind", "fast_descent",
                "lateral", "low_thrust", "recovery"
            ],
            "horizon": VALIDATION_HORIZON,
            "score_mode": "contest",
            "total_return": 26841.944,
            "landed": 169,
            "success_pct": 70.4167,
            "baseline_v3b_landed": 152,
            "baseline_v3b_return": 22475.215,
            "paired_mean_delta_return": 18.1947,
            "family_weighted_bootstrap_95_ci": [11.8034, 24.4092],
            "total_reward_bootstrap_95_ci": [2401.9704, 6475.3649],
            "landing_delta_vs_v3b": 17,
            "hull_contact_delta_vs_v3b": -2,
            "invalid_landing_delta_vs_v3b": -15
        },
        "smoke_test": {
            "observations": 4,
            "observation_shape": [4, 42],
            "logits_shape": [4, 9],
            "action_range": [0, 8],
            "simulator_steps": 0
        },
        "scope_note": (
            "Local six-family validation only. This manifest does not claim "
            "an official competition score or hidden-test result."
        )
    }

    manifest_text = json.dumps(
        manifest, ensure_ascii=False, indent=2
    ) + "\n"

    readme_text = f"""# Final submission candidate — Task 7

File: weights.pt

Model: {MODEL_NAME}
Architecture: 42 -> 32 -> 32 -> 9
Dtype: float32
SHA-256: {packaged_hash}

## Validation

- Seed: {VALIDATION_SEED}
- 240 matched medium episodes; 40 per each of six families
- Horizon: {VALIDATION_HORIZON}
- Successful landings: 169/240 (70.42%)
- Total reward: 26841.944
- Baseline v3b: 152/240 landings, reward 22475.215
- Paired mean reward delta: +18.1947 per episode
- Family-weighted bootstrap 95% CI: [11.8034, 24.4092]
- Paired total-reward bootstrap 95% CI: [2401.9704, 6475.3649]

The artifact was copied byte-for-byte from the checkpoint evaluated in
CELL 11B.10 and reloaded with torch_lander.load. A four-observation
inference smoke test passed. No simulator steps were executed during packaging.

This is a locally validated candidate. The validation metrics are not an
official competition score and do not guarantee performance on hidden tests.
See manifest.json for machine-readable provenance.
"""

    _write_immutable(output_dir / "manifest.json", manifest_text)
    _write_immutable(output_dir / "README.txt", readme_text)

    print("CELL 11B.11 — PACKAGE VALIDATION")
    print(f"Source:   {source}")
    print(f"Artifact: {target}")
    print(f"SHA-256:  {packaged_hash}")
    print("State dict: PASS (6 tensors, expected shapes, float32, finite)")
    print("Loader: PASS")
    print("Inference smoke test: PASS (4 x 42 -> 4 x 9)")
    print("Simulator steps: 0")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print(f"README:   {output_dir / 'README.txt'}")
    print("Original checkpoints were not modified.")
    print("QC: PASS")
    return target, output_dir / "manifest.json", output_dir / "README.txt"
