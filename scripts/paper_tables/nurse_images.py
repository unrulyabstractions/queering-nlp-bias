"""Paper images for the nurse (local bias) study, cut from the plots of scripts/visualize_estimation.py (default, not camera-ready).

  summary_core_evolution.png  <viz>/summary_core_evolution.png with the metadata footer blanked
  evolution_{root,trunk}.png  <viz>/log-inv-ppl/orientation/evolution_*.png, top 30 rows cropped, title remnant and footer blanked
  mc_{avg,greedy,mode,median}.png  the four 815x500 panels of <viz>/method_comparison.png (trunk), y-axis label blanked on the left two

Each PNG also gets a PDF sibling. Usage: uv run python scripts/paper_tables/nurse_images.py --viz out/nurse_sub/all/viz --out <paper>/images
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
    im.save(out / f"{name}.png"); im.save(out / f"{name}.pdf", resolution=300)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--viz", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    viz, out = Path(a.viz), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    s = np.array(Image.open(viz / "summary_core_evolution.png").convert("RGB"))
    blank_footer(s, rows=250, frac=0.45); save(s, out, "summary_core_evolution")
    for arm in ("root", "trunk"):
        e = np.array(Image.open(viz / "log-inv-ppl" / "orientation" / f"evolution_{arm}.png").convert("RGB"))[30:]
        e = e[: e.shape[0] - (e.shape[0] - 2429)] if e.shape[0] > 2429 else e
        white(e, 0, e.shape[1], 0, 46); white(e, 0, 2230, e.shape[0] - 170, e.shape[0])  # the footer text ends before the branch boxes at x = 2239
        save(e, out, f"evolution_{arm}")
    m = np.array(Image.open(viz / "method_comparison.png").convert("RGB"))
    for name, (x, y) in {"mc_avg": (0, 175), "mc_greedy": (815, 175), "mc_mode": (0, 755), "mc_median": (815, 755)}.items():
        p = m[y:y + 500, x:x + 815].copy()
        if x == 0: white(p, 0, 40, 60, 420)
        save(p, out, name)
    print("nurse images ->", out)

if __name__ == "__main__":
    main()
