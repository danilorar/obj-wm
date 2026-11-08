"""Shared utilities for trajectory model training and evaluation."""

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_default(obj):
    """Let json.dump handle numpy scalars in manifest fields."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def read_manifest(checkpoint):
    """Read the run settings saved beside a checkpoint."""
    return json.loads(Path(checkpoint).with_suffix(".json").read_text())


def save_run_manifest(checkpoint_path, **info) -> None:
    """Write a JSON manifest next to checkpoint_path recording how it was produced.

    Library versions are filled in automatically; pass everything else
    that matters for reproducing or auditing this checkpoint (e.g. seed,
    num_epochs, val_minade6/val_minfde6).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    manifest = {
        "checkpoint": str(checkpoint_path),
        "torch_version": torch.__version__,
        **info,
    }
    # Serialize before opening so invalid values cannot truncate an old manifest.
    payload = json.dumps(manifest, indent=2, default=_json_default, allow_nan=False)
    manifest_path = checkpoint_path.with_suffix(".json")
    manifest_path.write_text(payload + "\n")


def multimodal_metrics(predictions, targets):
    """Return per-sample minADE and minFDE, choosing modes independently.

    predictions: [B, K, 12, 2]; targets: [B, 12, 2].
    Each returned tensor has shape [B]. For K=6 these are minADE6/minFDE6.
    """
    if predictions.ndim != 4 or predictions.shape[1] < 1 or predictions.shape[2:] != (12, 2):
        raise ValueError("Expected predictions with shape [B, K, 12, 2], K >= 1")
    if targets.shape != (predictions.shape[0], 12, 2):
        raise ValueError("Expected targets with shape [B, 12, 2]")

    errors = torch.norm(predictions - targets[:, None], dim=-1)
    min_ade = errors.mean(dim=-1).min(dim=-1).values
    min_fde = errors[:, :, -1].min(dim=-1).values
    return min_ade, min_fde
