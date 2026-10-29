"""Which axes move: every axis is a score in [0,1]. An axis is shown in the paper's figures and per-axis tables when at
least one of the five group comparisons (Latino vs Polish, Latino vs formal, Polish vs formal, explicit vs implicit within
each group) differs by at least 0.2. Every axis stays in the catalogue."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

THRESHOLD = 0.2
LABELS = {"lat_pol": "Latino vs Polish", "lat_f": "Latino vs formal", "pol_f": "Polish vs formal",
          "lat_ei": "Latino explicit vs implicit", "pol_ei": "Polish explicit vs implicit"}
ORDER5 = ["lat_pol", "lat_f", "pol_f", "lat_ei", "pol_ei"]

def comparisons(analysis_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(analysis_dir) / "comparisons.csv")

def moving_axes(comp: pd.DataFrame, threshold: float = THRESHOLD) -> set[str]:
    w = comp.pivot(index="axis", columns="comparison", values="diff")
    return set(w.index[w.abs().max(axis=1) >= threshold])

def largest(comp: pd.DataFrame, axis: str):
    """(|difference|, comparison key) of the largest of the five comparisons for one axis, or (nan, None)."""
    g = comp[comp.axis == axis].dropna(subset=["diff"])
    if not len(g): return float("nan"), None
    r = g.loc[g["diff"].abs().idxmax()]
    return abs(float(r["diff"])), r["comparison"]
