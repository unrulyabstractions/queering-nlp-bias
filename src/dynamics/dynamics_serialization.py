"""Serialization for dynamics results."""

from __future__ import annotations

import json
from pathlib import Path

from src.common.file_io import ensure_dir

from .dynamics_types import DynamicsResult


def save_dynamics_json(result: DynamicsResult, output_path: Path | str) -> Path:
    """Save dynamics result to JSON.

    Format:
    {
        "n_structures": 4,
        "step": 4,
        "trajectories": [
            {
                "traj_idx": 0,
                "arm_name": "trunk",
                "n_tokens": 64,
                "pull": [[k, value], ...],
                "drift": [[k, value], ...],
                "potential": [[k, value], ...]
            }
        ]
    }
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    data = {
        "n_structures": result.n_structures,
        "step": result.step,
        "trajectories": [
            {
                "traj_idx": t.traj_idx,
                "arm_name": t.arm_name,
                "n_tokens": t.n_tokens,
                "pull": t.pull_series,
                "drift": t.drift_series,
                "potential": t.potential_series,
            }
            for t in result.trajectories
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path
