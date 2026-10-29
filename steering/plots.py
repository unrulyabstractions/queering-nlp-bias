"""Figures for the steering appendix.

Usage: uv run python steering/plots.py --run full [--top 12] [--out steering/out/<run>/figs]
Reads steering/out/<run>/analysis/{summary.csv, agreement.csv, table.csv}.
Palette (validated with the dataviz validator, light mode): baseline aqua #1baf7a, Latino blue #2a78d6,
Polish orange #eb6834; sequential blue ramp for the heatmap. Text in ink, never in series color.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, str(Path(__file__).resolve().parent))
from axes import AXES, JUDGE_AXES
from prompts import ORDER, ARMS

HERE = Path(__file__).resolve().parent
COL = {"none": "#1baf7a", "latino": "#2a78d6", "polish": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ARM_LABEL = {"F": "formal", "D": "Democrat", "EL": "explicit", "IL": "implicit", "HL": "heavy", "EP": "explicit", "IP": "implicit", "HP": "heavy"}
AX_NAME = {a["id"]: a["id"].replace("_", " ") for a in AXES}

plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300, "pdf.fonttype": 42})

def arm_color(arm): return COL[ARMS[arm]["group"]]

def group_brackets(ax, y=-0.40, short=False):
    """Label the three groups under the x axis."""
    spans = {"baseline": (0, 1), "Latino": (2, 4), ("Polish" if short else "Polish-American"): (5, 7)}
    for name, (a, b) in spans.items():
        ax.annotate("", xy=(a - 0.4, y), xytext=(b + 0.4, y), xycoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        ax.text((a + b) / 2, y - 0.04, name, ha="center", va="top", fontsize=7.5, color=INK2, transform=ax.get_xaxis_transform())

def bars(ax, summ, axis, ylabel=None, fmt="{:.2f}", ylim=None, title=None, labels=True, label_size=6):
    d = summ[summ.axis == axis].set_index("arm").reindex(ORDER)
    x = np.arange(len(ORDER))
    ax.bar(x, d["mean"], width=0.62, color=[arm_color(a) for a in ORDER], linewidth=0)
    yerr = np.vstack([(d["mean"] - d["lo"]).clip(lower=0), (d["hi"] - d["mean"]).clip(lower=0)])
    ax.errorbar(x, d["mean"], yerr=yerr, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels([ARM_LABEL[a] for a in ORDER], rotation=45, ha="right", fontsize=6.5)
    if labels:
        for xi, v, h in zip(x, d["mean"], d["hi"]):
            if not np.isnan(v): ax.text(xi, (h if not np.isnan(h) else v), (fmt(v) if callable(fmt) else fmt.format(v)), ha="center", va="bottom", fontsize=label_size, color=INK)
    if ylabel: ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    if title: ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ax.grid(axis="x", visible=False)

def fig_shares(summ, out):
    """One figure, six panels: the three census shares of the offered areas and three response shares, per arm."""
    fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.6))
    short = lambda v: "0" if v < 0.005 else ("1.0" if v > 0.995 else f"{v:.2f}"[1:])
    bars(axs[0, 0], summ, "hispanic_share_mean", ylabel="share of offered areas (%)", fmt="{:.0f}", title="Hispanic share", label_size=5.5)
    bars(axs[0, 1], summ, "white_nh_share_mean", fmt="{:.0f}", title="white non-Hispanic share", label_size=5.5)
    bars(axs[0, 2], summ, "black_nh_share_mean", fmt="{:.0f}", title="Black non-Hispanic share", label_size=5.5)
    bars(axs[1, 0], summ, "enclave_offered_latino_list", ylabel="share of responses", fmt=short, ylim=(0, 1.18), title="majority-Hispanic area offered", label_size=5.5)
    bars(axs[1, 1], summ, "enclave_offered_polish_list", fmt=short, ylim=(0, 1.18), title="Polish-heritage area offered", label_size=5.5)
    bars(axs[1, 2], summ, "l2_greeting_signoff", fmt=short, ylim=(0, 1.18), title="Spanish or Polish greeting", label_size=5.5)
    top = max(ax.get_ylim()[1] for ax in axs[0]);
    for ax in axs[0]: ax.set_ylim(0, top)
    for ax in axs.flat: group_brackets(ax, short=True)
    fig.tight_layout(h_pad=2.6, w_pad=1.0)
    fig.savefig(out / "steering-shares.png"); fig.savefig(out / "steering-shares.pdf"); plt.close(fig)

def fig_core_grid(summ, out, judge_ids):
    """Default behavior per user group as one figure: four rows by two columns of vertical bar panels over the headline axes.
    Row one holds the two baseline arms; rows two to four hold explicit, implicit and heavy, Mexican-American left, Polish-American right."""
    fam = {a["id"]: a["family"] for a in AXES}
    rows = [(k, lab, "steering") for k, lab in CORE_SCRIPT]
    for f in FAM_ORDER[1:]:
        rows += [(k, AX_NAME.get(k, k), f) for k in judge_ids if fam.get(k) == f]
    x = np.arange(len(rows))
    grid = [["F", "D"], ["EL", "EP"], ["IL", "IP"], ["HL", "HP"]]
    fig, axs = plt.subplots(4, 2, figsize=(6.6, 7.4), sharex=True, sharey=True)
    for r, pair in enumerate(grid):
        for c, arm in enumerate(pair):
            ax = axs[r, c]; d = summ[summ.arm == arm].set_index("axis")
            vals = np.array([d["mean"].get(k, np.nan) for k, _, _ in rows], dtype=float)
            lo = np.array([d["lo"].get(k, np.nan) for k, _, _ in rows], dtype=float); hi = np.array([d["hi"].get(k, np.nan) for k, _, _ in rows], dtype=float)
            ax.bar(x, vals, width=0.68, color=[FAM_COL[f] for _, _, f in rows], linewidth=0)
            ok = ~np.isnan(lo)
            ax.errorbar(x[ok], vals[ok], yerr=np.vstack([(vals - lo)[ok].clip(0), (hi - vals)[ok].clip(0)]), fmt="none", ecolor=INK2, elinewidth=0.6, capsize=1.2)
            ax.axhline(0.5, color=INK2, lw=0.6, ls=(0, (3, 3)))
            ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", ".5", "1"], fontsize=6)
            ax.set_title(ARM_TITLE[arm], fontsize=8, color=INK, loc="left", pad=3)
            ax.grid(visible=False)
            for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
            ax.tick_params(axis="x", length=2, color=GRID); ax.tick_params(axis="y", length=0)
    for ax in axs[-1]:
        ax.set_xticks(x); ax.set_xticklabels([lab for _, lab, _ in rows], rotation=45, ha="right", fontsize=6, rotation_mode="anchor")
    used = [f for f in FAM_ORDER if any(r[2] == f for r in rows)]
    names = {"steering": "recommended set (script)", "identity": "identity", "constraints": "constraints", "preferences": "preferences", "form": "form"}
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAM_COL[f]) for f in used]
    fig.legend(handles, [names[f] for f in used], loc="upper center", ncol=len(used), fontsize=6.5, bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.2, w_pad=1.0)
    fig.savefig(out / "steering-core-grid.png"); fig.savefig(out / "steering-core-grid.pdf"); plt.close(fig)

def fig_axes_grid(summ, out, axes_ids, fname="steering-axes", ncols=4, ylabel="ensemble yes rate"):
    n = len(axes_ids); nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.6, 1.95 * nrows + 0.3), squeeze=False)
    for i, axis in enumerate(axes_ids):
        ax = axs[i // ncols][i % ncols]
        import textwrap
        bars(ax, summ, axis, ylabel=ylabel if i % ncols == 0 else None, fmt="{:.2f}", ylim=(0, 1.12),
             title="\n".join(textwrap.wrap(AX_NAME.get(axis, axis), 22)), labels=False)
        ax.tick_params(axis="x", labelsize=5.5)
    for j in range(n, nrows * ncols): axs[j // ncols][j % ncols].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[g]) for g in ("none", "latino", "polish")]
    fig.legend(handles, ["baseline (formal, Democrat)", "Latino arms", "Polish-American arms"], loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / f"{fname}.png"); fig.savefig(out / f"{fname}.pdf"); plt.close(fig)

FAM_COL = {"steering": "#4b5d78", "identity": "#7b4fb3", "constraints": "#1f8a8a", "preferences": "#c9972b", "form": "#8a6d5a"}
FAM_ORDER = ["steering", "identity", "constraints", "preferences", "form"]
CORE_SCRIPT = [("enclave_offered_latino_list", "majority-Hispanic area offered"), ("enclave_offered_polish_list", "Polish-heritage area offered"),
               ("commute_fail_share", "offered areas beyond commute limit"), ("l2_greeting_signoff", "Spanish or Polish greeting")]

def fig_cores(summ, out, judge_ids, fname="steering-cores"):
    """Default behavior per arm: one panel per arm, one horizontal bar per axis, dashed midline, as the paper's core bars."""
    fam = {a["id"]: a["family"] for a in AXES}
    rows = [(k, lab, "steering") for k, lab in CORE_SCRIPT]
    for f in FAM_ORDER[1:]:
        rows += [(k, AX_NAME.get(k, k), f) for k in judge_ids if fam.get(k) == f]
    n = len(rows)
    fig, axs = plt.subplots(1, len(ORDER), figsize=(6.8, 0.17 * n + 1.1), sharey=True)
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
    axs[0].set_yticks(y); axs[0].set_yticklabels([lab for _, lab, _ in rows], fontsize=6)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93)); fig.subplots_adjust(wspace=0.18)
    # group labels above the panels, in figure coordinates
    spans = {"baseline": (0, 1), "Latino": (2, 4), "Polish-American": (5, 7)}
    for name, (a, b) in spans.items():
        x0 = axs[a].get_position().x0; x1 = axs[b].get_position().x1; ytop = axs[a].get_position().y1 + 0.055
        fig.add_artist(plt.Line2D([x0, x1], [ytop, ytop], color=INK2, lw=0.8))
        fig.text((x0 + x1) / 2, ytop + 0.012, name, ha="center", va="bottom", fontsize=7.5, color=INK2)
    used = [f for f in FAM_ORDER if any(r[2] == f for r in rows)]
    names = {"steering": "recommended set (script)", "identity": "identity", "constraints": "constraints", "preferences": "preferences", "form": "form"}
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAM_COL[f]) for f in used]
    fig.legend(handles, [names[f] for f in used], loc="lower center", ncol=len(used), fontsize=6.5, bbox_to_anchor=(0.5, 0.0), handlelength=1.2, columnspacing=1.2)
    fig.savefig(out / f"{fname}.png"); fig.savefig(out / f"{fname}.pdf"); plt.close(fig)

ARM_TITLE = {"F": "formal, no identity", "D": "formal, \u201cI am a Democrat\u201d", "EL": "Mexican-American, explicit", "IL": "Mexican-American, implicit",
             "HL": "Mexican-American, heavy", "EP": "Polish-American, explicit", "IP": "Polish-American, implicit", "HP": "Polish-American, heavy"}

def fig_heatmap(summ, out, axes_ids, fname="steering-heatmap"):
    piv = summ[summ.axis.isin(axes_ids)].pivot(index="axis", columns="arm", values="mean").reindex(axes_ids)[ORDER]
    piv = piv[piv.max(axis=1) >= 0.05]; axes_ids = list(piv.index)
    fig, ax = plt.subplots(figsize=(6.6, 0.145 * len(axes_ids) + 0.9))
    im = ax.imshow(piv.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([f"{a}\n{ARM_LABEL[a]}" for a in ORDER], fontsize=7)
    ax.set_yticks(range(len(axes_ids))); ax.set_yticklabels([AX_NAME.get(a, a) for a in axes_ids], fontsize=6.5)
    ax.grid(False)
    for i in range(len(axes_ids)):
        for j in range(len(ORDER)):
            v = piv.values[i, j]
            if not np.isnan(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color=("white" if v > 0.6 else INK))
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label("yes rate", fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(out / f"{fname}.png"); fig.savefig(out / f"{fname}.pdf"); plt.close(fig)

def fig_agreement(agree, out):
    a = agree[(agree.yes_rate_overall >= 0.02) & (agree.yes_rate_overall <= 0.98) & ~agree.axis.str.endswith("@r1")].sort_values("fleiss_kappa")
    fig, ax = plt.subplots(figsize=(6.6, 0.13 * len(a) + 0.8))
    y = np.arange(len(a))
    ax.hlines(y, 0, a.fleiss_kappa, color=GRID, lw=1); ax.plot(a.fleiss_kappa, y, "o", ms=3.5, color="#2a78d6")
    ax.plot(a.pairwise_agreement, y, "s", ms=3, color="#eb6834")
    ax.set_yticks(y); ax.set_yticklabels([AX_NAME.get(x, x) for x in a.axis], fontsize=6.5)
    ax.set_xlabel("Fleiss kappa (circles) and pairwise agreement (squares) across the three judges"); ax.set_xlim(-0.2, 1.02)
    ax.legend(handles=[plt.Line2D([], [], marker="o", color="#2a78d6", ls=""), plt.Line2D([], [], marker="s", color="#eb6834", ls="")],
              labels=["Fleiss kappa", "pairwise agreement"], loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(out / "steering-agreement.png"); fig.savefig(out / "steering-agreement.pdf"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default=None); args = ap.parse_args()
    run_dir = HERE / "out" / args.run; an = run_dir / "analysis"
    out = Path(args.out) if args.out else run_dir / "figs"; out.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(an / "summary.csv"); agree = pd.read_csv(an / "agreement.csv")
    judge_ids = [a["id"] for a in JUDGE_AXES if a["id"] in set(summ.axis)]
    piv = summ[summ.axis.isin(judge_ids)].pivot(index="axis", columns="arm", values="mean")
    ident = [c for c in piv.columns if c not in ("F", "D")]
    gap = pd.concat([(piv[ident].max(axis=1) - piv["F"]).abs(), (piv[ident].min(axis=1) - piv["F"]).abs()], axis=1).max(axis=1)
    top = gap.sort_values(ascending=False).index.tolist()
    fig_shares(summ, out); fig_cores(summ, out, top[:16]); fig_core_grid(summ, out, top[:16])
    fig_axes_grid(summ, out, top[: args.top]); fig_heatmap(summ, out, top)
    if len(agree): fig_agreement(agree, out)
    json.dump({"top_axes": top[: args.top], "gap": gap.sort_values(ascending=False).round(3).to_dict()}, (an / "top_axes.json").open("w"), indent=1)
    print("figures ->", out, "| top axes:", top[: args.top])

if __name__ == "__main__":
    main()
