"""Which axes move: every axis is a score in [0,1]; an axis is shown in the paper's figures and per-axis tables when its
per-arm mean ranges by at least 0.2 across the eight arms. Every axis stays in the catalogue."""
from __future__ import annotations
import pandas as pd

THRESHOLD = 0.2

def moving_axes(summ: pd.DataFrame, threshold: float = THRESHOLD) -> set[str]:
    piv = summ.pivot(index="axis", columns="arm", values="mean")
    rng = piv.max(axis=1) - piv.min(axis=1)
    return set(rng[rng >= threshold].index)
