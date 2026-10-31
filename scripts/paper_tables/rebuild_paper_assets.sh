#!/bin/bash
# Regenerate every table and image the paper takes from this repo. No API calls: it reads the stored judge verdicts,
# and the scoring step runs with --offline, so a missing reply stops the script instead of calling a judge.
# Usage: bash scripts/paper_tables/rebuild_paper_assets.sh [paper-repo]
set -euo pipefail
cd "$(dirname "$0")/../.."
PAPER="${1:-$HOME/work/papers/homo}"
JUDGES="opus gpt5 gemini haiku55 qwen deepseek"
ROOT=out/nurse_sub

# Local bias: per-judge files, the ensemble, estimates, plots.
for j in haiku55 qwen deepseek; do uv run python scripts/score_nurse_extra_judges.py --judge "$j" --offline | grep -E "^done"; done
for j in $JUDGES; do uv run python scripts/estimate_normativity.py "$ROOT/$j/scoring.json" > /dev/null; done
rm -rf "$ROOT/all/viz"
uv run python scripts/merge_judge_runs.py --gen-dir "$ROOT" --judges $JUDGES --name all > /dev/null
uv run python scripts/estimate_normativity.py "$ROOT/all/scoring.json" > /dev/null
VIZ_HIDE_METADATA=1 uv run python scripts/visualize_estimation.py "$ROOT/all/estimation.json" > /dev/null

# Local bias: tables and images.
uv run python scripts/paper_tables/nurse_judge_tables.py --judges $JUDGES --ensemble all --out "$PAPER/appendix/generated"
uv run python steering/judges_local.py --judges $JUDGES --tex "$PAPER/appendix/generated/judges" --img "$PAPER/images/judges" > /dev/null
uv run python scripts/paper_tables/nurse_tables.py --out "$PAPER/appendix/generated" --judges $JUDGES --per-arm 4 --female 5 --width 7.5cm
uv run python scripts/paper_tables/nurse_images.py --viz "$ROOT/all/viz" --out "$PAPER/images"

# Global bias: the three arms the paper reports.
export STEERING_ARMS=F,EL,IL
uv run python steering/plots.py --run full --out "$PAPER/images"
uv run python steering/tables_tex.py --run full --out "$PAPER/appendix/generated"
echo "paper assets rebuilt -> $PAPER"
