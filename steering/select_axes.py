"""Which axes move: an axis is shown in the paper's figures and per-axis tables when its per-arm mean ranges by at
least 0.2 across the eight arms (axes in [0,1]), or by at least 20 percent of its largest absolute value (counts,
census percentages, rents). Every axis stays in the catalogue."""
from __future__ import annotations
import numpy as np, pandas as pd

THRESHOLD = 0.2

def moving_axes(summ: pd.DataFrame, threshold: float = THRESHOLD) -> set[str]:
    piv = summ.pivot(index="axis", columns="arm", values="mean")
    rng = piv.max(axis=1) - piv.min(axis=1)
    unit = piv.apply(lambda r: r.dropna().between(-1e-9, 1 + 1e-9).all(), axis=1)
    rel = (rng / piv.abs().max(axis=1).replace(0, np.nan)).fillna(0)
    keep = set(rng[unit & (rng >= threshold)].index) | set(rel[~unit & (rel >= threshold)].index)
    return keep
