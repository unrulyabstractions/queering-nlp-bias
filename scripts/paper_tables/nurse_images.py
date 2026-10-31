"""Paper images for the nurse (local bias) study, cut from the plots of scripts/visualize_estimation.py run with VIZ_HIDE_METADATA=1.

  local-bias/default-behavior-tree.png    <viz>/summary_core_evolution.png
  local-bias/orientation-tree-{root,trunk}.png  <viz>/log-inv-ppl/orientation/evolution_*.png, top 30 rows cropped, title remnant blanked
  foundations/statistic-{average,greedy,mode,median}.png  the four panels of <viz>/method_comparison.png (trunk), y-axis label blanked on the left two

Usage: uv run python scripts/paper_tables/nurse_images.py --viz out/nurse_sub/all/viz --out <paper>/images
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def white(a, x0, x1, y0, y1): a[max(0, y0):y1, max(0, x0):x1] = 255

def blank_footer(a, rows=130, frac=0.6):
    """White out the bottom-left metadata footer (Method / Judge LLM / Gen Model lines)."""
    h, w = a.shape[:2]; white(a, 0, int(w * frac), h - rows, h)

def save(a, out: Path, name: str):
    im = Image.fromarray(a.astype(np.uint8)).convert("RGB")
    (out / name).parent.mkdir(parents=True, exist_ok=True); im.save(out / f"{name}.png")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--viz", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    viz, out = Path(a.viz), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    s = np.array(Image.open(viz / "summary_core_evolution.png").convert("RGB"))
    blank_footer(s, rows=250, frac=0.45); save(s, out, "local-bias/default-behavior-tree")
    for arm in ("root", "trunk"):
        e = np.array(Image.open(viz / "log-inv-ppl" / "orientation" / f"evolution_{arm}.png").convert("RGB"))[30:]
        e = e[: e.shape[0] - (e.shape[0] - 2429)] if e.shape[0] > 2429 else e
        white(e, 0, e.shape[1], 0, 46); white(e, 0, 2230, e.shape[0] - 170, e.shape[0])  # the footer text ends before the branch boxes at x = 2239
        save(e, out, f"local-bias/orientation-tree-{arm}")
    m = np.array(Image.open(viz / "method_comparison.png").convert("RGB"))
    # Crop both rows at the same height relative to their dashed 0.5 line, so the two rows show the same y-range
    # (the bottom row used to be cut 35 px too low, clipping the "1.00" labels).
    near = (np.abs(m[:, 90:800].astype(int) - 204).sum(2) < 12).sum(1)
    lines = [y for y in range(m.shape[0]) if near[y] > 150]
    assert len(lines) == 2, lines
    y2 = 175 + (lines[1] - lines[0])
    # The two panel columns are separated only by the blank columns 842-843: crop at that gap so no panel carries a
    # sliver of its neighbour, and pad the right panels to the same width so all four scale to the same height.
    ink = (m[150:1290].astype(int).min(2) < 235).sum(0)
    gap = next(x for x in range(760, 950) if ink[x] == 0)
    W = gap
    for name, (x, y) in {"average": (0, 175), "greedy": (gap + 1, 175), "mode": (0, y2), "median": (gap + 1, y2)}.items():
        p = m[y:y + 500, x:min(x + W, m.shape[1])].copy()
        if p.shape[1] < W: p = np.concatenate([p, np.full((p.shape[0], W - p.shape[1], 3), 255, dtype=p.dtype)], axis=1)
        if x == 0: white(p, 0, 40, 60, 420)
        save(p, out, f"foundations/statistic-{name}")
    print("nurse images ->", out)

if __name__ == "__main__":
    main()
