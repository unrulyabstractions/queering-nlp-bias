"""Figures for the global-bias appendix.

Usage: STEERING_ARMS=F,EL,IL uv run python steering/plots.py --run full --out <paper>/images
Reads steering/out/<run>/<ANALYSIS_DIR>/{summary,agreement,table,comparisons}.csv.
Writes global-bias/{arms-all,heatmap,comparison-formal,comparison-explicit-implicit}.png and judges/global-agreement.png.
Sequential blue ramp for the heatmap, validated with the dataviz validator in light mode. Text in ink, never in series color.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, str(Path(__file__).resolve().parent))
from axes import AXES, JUDGE_AXES
from select_axes import moving_axes, comparisons
from prompts import ORDER, ANALYSIS_DIR
_POL = bool({'EP', 'IP', 'HP'} & set(ORDER))
L2_GREET = 'Spanish or Polish greeting' if _POL else 'Spanish greeting'
L2_SWITCH = 'reply in Spanish or Polish' if _POL else 'reply in Spanish'

HERE = Path(__file__).resolve().parent
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ARM_LABEL = {"F": "formal", "D": "Democrat", "EL": "explicit", "IL": "implicit", "HL": "heavy", "EP": "explicit", "IP": "implicit", "HP": "heavy"}
AX_NAME = {a["id"]: a["id"].replace("_", " ") for a in AXES}

plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300, "pdf.fonttype": 42})

FAM_COL = {"steering": "#4b5d78", "identity": "#7b4fb3", "constraints": "#1f8a8a", "preferences": "#c9972b", "service": "#b5536b", "form": "#8a6d5a"}
FAM_ORDER = ["steering", "identity", "constraints", "preferences", "service", "form"]
CORE_SCRIPT = [("enclave_offered_latino_list", "majority-Hispanic area offered"), ("enclave_offered_polish_list", "Polish-heritage area offered"),
               ("commute_fail_share", "offered areas beyond commute limit"), ("over_limit_share", "offered areas the reply puts over 30 min"),
               ("adjacent_to_imd_offered", "walk-to-work area offered"), ("l2_greeting_signoff", L2_GREET)]

def all_rows(summ):
    """Every judge axis and every script axis that lives in [0,1], as (id, label, family) rows: families in FAM_ORDER,
    within a family by the largest gap from the baseline, so the eye meets the moving axes first."""
    fam = {a["id"]: a["family"] for a in AXES}
    label = dict(CORE_SCRIPT); label.update({"westside_black_area_named": "majority-Black West Side area offered", "full_switch": L2_SWITCH,
                                            "congrats_opener": "opens with congratulations", "identity_in_opening": "ethnicity named in the opening",
                                            "top_pick_over_limit": "closing pick over the commute limit", "enclave_epithet": "ethnic nickname for the enclave",
                                            "offrail_share": "offered areas without rail", "rent_over_budget_share": "offered areas over budget",
                                            "polish_heritage_share": "share of offered areas Polish-heritage", "family_section_share": "share of words on the family",
                                            "hispanic_share_mean": "Hispanic share of offered areas", "white_nh_share_mean": "white share of offered areas",
                                            "black_nh_share_mean": "Black share of offered areas"})
    judge = {a["id"] for a in JUDGE_AXES}
    ids = [a["id"] for a in JUDGE_AXES if a["id"] in set(summ.axis)] + [k for k in sorted(set(summ.axis)) if k not in judge and "@" not in k]
    for k in ids: fam.setdefault(k, "steering")
    piv = summ[summ.axis.isin(ids)].pivot(index="axis", columns="arm", values="mean")
    ident = [c for c in piv.columns if c not in ("F", "D")]
    gap = pd.concat([(piv[ident].max(axis=1) - piv["F"]).abs(), (piv[ident].min(axis=1) - piv["F"]).abs()], axis=1).max(axis=1)
    rows = []
    for f in FAM_ORDER:
        ks = sorted([k for k in ids if fam.get(k) == f], key=lambda k: -gap.get(k, 0))
        rows += [(k, label.get(k, AX_NAME.get(k, k)), f) for k in ks]
    return rows

def fig_cores(summ, out, rows, row_h, label_size):
    """Default behavior per arm: one panel per arm, one horizontal bar per axis, dashed midline, as the paper's core bars."""
    n = len(rows)
    fig, axs = plt.subplots(1, len(ORDER), figsize=(6.8, row_h * n + 1.1), sharey=True)
    y = np.arange(n)[::-1]
    for ax, arm in zip(axs, ORDER):
        d = summ[summ.arm == arm].set_index("axis")
        vals = np.array([d["mean"].get(k, np.nan) for k, _, _ in rows], dtype=float)
        lo = np.array([d["lo"].get(k, np.nan) for k, _, _ in rows], dtype=float); hi = np.array([d["hi"].get(k, np.nan) for k, _, _ in rows], dtype=float)
        ax.barh(y, vals, height=0.62, color=[FAM_COL[f] for _, _, f in rows], linewidth=0)
        ok = ~np.isnan(lo)
        ax.errorbar(vals[ok], y[ok], xerr=np.vstack([(vals - lo)[ok].clip(0), (hi - vals)[ok].clip(0)]), fmt="none", ecolor=INK2, elinewidth=0.5, capsize=0)
        ax.axvline(0.5, color=INK2, lw=0.6, ls=(0, (3, 3)))
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", ".5", "1"], fontsize=5.5)
        ax.set_title(ARM_LABEL[arm], fontsize=7.5, color=INK, loc="center", pad=3)
        ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
        ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(y); axs[0].set_yticklabels([lab for _, lab, _ in rows], fontsize=label_size)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93)); fig.subplots_adjust(wspace=0.18)
    # group labels above the panels, in figure coordinates
    grp = {"F": "baseline", "D": "baseline", "EL": "Mexican-American", "IL": "Mexican-American", "HL": "Mexican-American",
           "EP": "Polish-American", "IP": "Polish-American", "HP": "Polish-American"}
    spans = {}
    for i, a in enumerate(ORDER): spans.setdefault(grp[a], [i, i])[1] = i
    spans = {k: tuple(v) for k, v in spans.items()}
    for name, (a, b) in spans.items():
        x0 = axs[a].get_position().x0; x1 = axs[b].get_position().x1; ytop = axs[a].get_position().y1 + 0.055
        fig.add_artist(plt.Line2D([x0, x1], [ytop, ytop], color=INK2, lw=0.8))
        fig.text((x0 + x1) / 2, ytop + 0.012, name, ha="center", va="bottom", fontsize=7.5, color=INK2)
    used = [f for f in FAM_ORDER if any(r[2] == f for r in rows)]
    names = {"steering": "recommended set", "identity": "identity", "constraints": "constraints", "preferences": "preferences", "service": "service", "form": "form"}
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAM_COL[f]) for f in used]
    fig.legend(handles, [names[f] for f in used], loc="lower center", ncol=len(used), fontsize=6.5, bbox_to_anchor=(0.5, 0.0), handlelength=1.2, columnspacing=1.2)
    fig.savefig(out / "arms-all.png"); plt.close(fig)

CMP_GROUPS = {"lat_f": (("Mexican-American (" + ", ".join(a for a in ["EL", "IL", "HL"] if a in ORDER) + ")", [a for a in ["EL", "IL", "HL"] if a in ORDER]), ("formal", ["F"])),
              "pol_f": (("Polish-American (EP, IP, HP)", ["EP", "IP", "HP"]), ("formal", ["F"])),
              "lat_pol": (("Latino (EL, IL, HL)", ["EL", "IL", "HL"]), ("Polish-American (EP, IP, HP)", ["EP", "IP", "HP"])),
              "lat_ei": (("Mexican-American explicit (EL)", ["EL"]), ("Mexican-American implicit (IL)", ["IL"])),
              "pol_ei": (("Polish explicit (EP)", ["EP"]), ("Polish implicit (IP)", ["IP"]))}

CMP_FILE = {"lat_f": "comparison-formal", "lat_ei": "comparison-explicit-implicit"}

def fig_comparisons(an, out, labels):
    """One core-bar figure per group comparison: two panels (the two groups), one bar per axis whose pooled means differ by at
    least 0.2, axes grouped by family and ordered by the difference; bootstrap intervals over responses."""
    table = pd.read_csv(an / "table.csv"); comp = pd.read_csv(an / "comparisons.csv")
    fam = {a["id"]: a["family"] for a in AXES}
    rng = np.random.default_rng(0)
    def mean_ci(x):
        x = x[~np.isnan(x)]
        if len(x) == 0: return np.nan, np.nan, np.nan
        bs = rng.choice(x, size=(1000, len(x)), replace=True).mean(axis=1)
        return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    for k, ((na, ga), (nb, gb)) in CMP_GROUPS.items():
        if k not in set(comp.comparison): continue
        g = comp[(comp.comparison == k) & (comp["diff"].abs() >= 0.2) & comp.axis.isin(moving_axes(comp, analysis_dir=an))].copy()
        g["fam"] = g.axis.map(lambda a: fam.get(a, "steering"))
        g["fo"] = g.fam.map(lambda f: FAM_ORDER.index(f) if f in FAM_ORDER else 0)
        g = g.sort_values(["fo", "diff"], ascending=[True, False])
        rows = [(r.axis, labels.get(r.axis, AX_NAME.get(r.axis, r.axis)), r.fam) for r in g.itertuples()]
        n = len(rows); y = np.arange(n)[::-1]
        fig, axs = plt.subplots(1, 2, figsize=(6.4, 0.2 * n + 1.3), sharey=True)
        for ax, (name, arms) in zip(axs, [(na, ga), (nb, gb)]):
            sub = table[table.arm.isin(arms)]
            st = [mean_ci(sub[a].astype(float).values) if a in sub else (np.nan,) * 3 for a, _, _ in rows]
            vals = np.array([t[0] for t in st]); lo = np.array([t[1] for t in st]); hi = np.array([t[2] for t in st])
            ax.barh(y, vals, height=0.62, color=[FAM_COL.get(f, FAM_COL["steering"]) for _, _, f in rows], linewidth=0)
            ok = ~np.isnan(lo)
            ax.errorbar(vals[ok], y[ok], xerr=np.vstack([(vals - lo)[ok].clip(0), (hi - vals)[ok].clip(0)]), fmt="none", ecolor=INK2, elinewidth=0.6, capsize=0)
            ax.axvline(0.5, color=INK2, lw=0.6, ls=(0, (3, 3)))
            ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", ".5", "1"], fontsize=6.5)
            ax.set_title(name, fontsize=8, color=INK, pad=4)
            ax.grid(visible=False)
            for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
            ax.tick_params(axis="y", length=0)
        axs[0].set_yticks(y); axs[0].set_yticklabels([lab for _, lab, _ in rows], fontsize=6.5)
        used = [f for f in FAM_ORDER if any(r[2] == f for r in rows)]
        names = {"steering": "recommended set", "identity": "identity", "constraints": "constraints", "preferences": "preferences", "service": "service", "form": "form"}
        fig.legend([plt.Rectangle((0, 0), 1, 1, color=FAM_COL[f]) for f in used], [names[f] for f in used], loc="lower center",
                   ncol=len(used), fontsize=6.5, bbox_to_anchor=(0.5, 0.0), handlelength=1.2, columnspacing=1.2)
        fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.subplots_adjust(wspace=0.08)
        fig.savefig(out / f"{CMP_FILE.get(k, 'comparison-' + k.replace('_', '-'))}.png"); plt.close(fig)

def fig_heatmap(summ, out, axes_ids, labels):
    piv = summ[summ.axis.isin(axes_ids)].pivot(index="axis", columns="arm", values="mean").reindex(axes_ids)[ORDER]
    piv = piv[piv.max(axis=1) >= 0.05]; axes_ids = list(piv.index)
    fig, ax = plt.subplots(figsize=(6.6, 0.118 * len(axes_ids) + 0.9))
    im = ax.imshow(piv.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([f"{a}\n{ARM_LABEL[a]}" for a in ORDER], fontsize=7)
    ax.set_yticks(range(len(axes_ids))); ax.set_yticklabels([labels.get(a, AX_NAME.get(a, a)) for a in axes_ids], fontsize=5.6)
    ax.grid(False)
    for i in range(len(axes_ids)):
        for j in range(len(ORDER)):
            v = piv.values[i, j]
            if not np.isnan(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.6, color=("white" if v > 0.6 else INK))
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label("mean score", fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(out / "heatmap.png"); plt.close(fig)

def fig_agreement(agree, out):
    a = agree[~agree.axis.str.contains("@")].dropna(subset=["fleiss_kappa"]).sort_values("fleiss_kappa")
    fig, ax = plt.subplots(figsize=(6.6, 0.11 * len(a) + 0.9))
    y = np.arange(len(a))
    ax.hlines(y, 0, a.fleiss_kappa, color=GRID, lw=1); ax.plot(a.fleiss_kappa, y, "o", ms=3.5, color="#2a78d6")
    ax.plot(a.pairwise_agreement, y, "s", ms=3, color="#eb6834")
    ax.set_yticks(y); ax.set_yticklabels([AX_NAME.get(x, x) for x in a.axis], fontsize=6.5)
    ax.set_xlabel("Fleiss kappa (circles) and pairwise agreement (squares) across the three judges"); ax.set_xlim(-0.2, 1.02)
    ax.axvline(0.5, color="#b5536b", lw=1.0, ls=(0, (4, 2)))
    ax.text(0.505, len(a) - 0.6, "kappa = 0.5", color="#b5536b", fontsize=6.5, ha="left", va="top")
    ax.legend(handles=[plt.Line2D([], [], marker="o", color="#2a78d6", ls=""), plt.Line2D([], [], marker="s", color="#eb6834", ls="")],
              labels=["Fleiss kappa", "pairwise agreement"], loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(out / "global-agreement.png"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", default=None); args = ap.parse_args()
    run_dir = HERE / "out" / args.run; an = run_dir / ANALYSIS_DIR
    out = Path(args.out) if args.out else run_dir / "figs"
    global_dir, judges_dir = out / "global-bias", out / "judges"
    global_dir.mkdir(parents=True, exist_ok=True); judges_dir.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(an / "summary.csv"); agree = pd.read_csv(an / "agreement.csv")
    keep = moving_axes(comparisons(an), analysis_dir=an)
    rows = [r for r in all_rows(summ) if r[0] in keep]
    fig_cores(summ, global_dir, rows, row_h=0.105, label_size=5.2)
    fig_comparisons(an, global_dir, {k: lab for k, lab, _ in all_rows(summ)})
    fig_heatmap(summ, global_dir, [k for k, _, _ in rows], {k: lab for k, lab, _ in rows})
    if len(agree): fig_agreement(agree, judges_dir)
    print("figures ->", out, "|", len(rows), "axes")

if __name__ == "__main__":
    main()
