"""Judge agreement for the local-bias (nurse love story) study, in the same form as the global study's agreement table and figure.

Reads out/nurse_sub/{opus,gpt5,gemini}/scoring.json (per-judge 0/1 verdicts per trajectory, arm and structure question).
Usage: uv run python steering/judges_local.py --tex <paper>/appendix/generated/judges --img <paper>/images
"""
from __future__ import annotations
import argparse, json, datetime, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import fleiss_kappa
import plots  # palette and matplotlib style

ROOT = Path(__file__).resolve().parent.parent / "out" / "nurse_sub"
JUDGES = [("opus", "Opus 4.6"), ("gpt5", "GPT-5"), ("gemini", "Gemini 2.5 Flash")]
AXES = ["male nurse", "female nurse", "heterosexual couple", "same-sex couple"]
ARMS = ["root", "trunk", "branch_1", "branch_2", "branch_3"]

def load():
    rows = []
    for key, _ in JUDGES:
        d = json.load(open(ROOT / key / "scoring.json"))
        for r in d["results"]:
            for q, v in enumerate(r["method_scores"]["categorical"]):
                rows.append({"judge": key, "arm": r["arm"], "traj": r["traj_idx"], "axis": AXES[q], "vote": v})
    df = pd.DataFrame(rows)
    return df.pivot_table(index=["arm", "traj", "axis"], columns="judge", values="vote").reset_index()

def agreement(w, groups):
    out = []
    for keys, g in w.groupby(groups):
        m = g[[k for k, _ in JUDGES]].dropna().values.astype(int)
        if not len(m): continue
        pair = np.mean([np.mean(m[:, i] == m[:, k]) for i in range(3) for k in range(i + 1, 3)])
        keys = keys if isinstance(keys, tuple) else (keys,)
        out.append({**dict(zip(groups, keys)), "n": len(m), "kappa": fleiss_kappa(m), "agree": pair,
                    **{f"yes_{k}": m[:, i].mean() for i, (k, _) in enumerate(JUDGES)}})
    return pd.DataFrame(out)

def f2(v): return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tex", required=True); ap.add_argument("--img", required=True); a = ap.parse_args()
    tex, img = Path(a.tex), Path(a.img); tex.mkdir(parents=True, exist_ok=True)
    w = load()
    by_axis = agreement(w, ["axis"]); by_cell = agreement(w, ["axis", "arm"])
    head = ("% Source: OLD/queering-nlp-bias/out/nurse_sub/{opus,gpt5,gemini}/scoring.json, per-judge 0/1 verdicts;\n"
            f"%         written by steering/judges_local.py on {datetime.date.today().isoformat()}. Generated, do not edit by hand.\n")
    L = [head, "\\begin{table}[h!]", "\\centering\\footnotesize",
         "\\caption{Judge agreement in the local-bias study: Fleiss' $\\kappa$ and pairwise agreement across the three judges, and each judge's yes rate, per structure question and arm (200 trajectories per arm). $\\kappa$ is undefined where every judge gives the same answer to every trajectory.}",
         "\\label{tab:judges-local-agreement}", "\\begin{tabular}{llrrrrr}", "\\toprule",
         "\\textbf{question} & \\textbf{arm} & $\\kappa$ & \\textbf{agree} & \\textbf{Opus 4.6} & \\textbf{GPT-5} & \\textbf{Gemini 2.5 Flash} \\\\", "\\midrule"]
    for ax in AXES:
        r = by_axis[by_axis.axis == ax].iloc[0]
        L.append(f"\\textbf{{{ax}}} & all & {f2(r.kappa)} & {f2(r.agree)} & {f2(r.yes_opus)} & {f2(r.yes_gpt5)} & {f2(r.yes_gemini)} \\\\")
        for arm in ARMS:
            g = by_cell[(by_cell.axis == ax) & (by_cell.arm == arm)]
            if not len(g): continue
            r = g.iloc[0]
            L.append(f" & {arm.replace('_', '\\_')} & {f2(r.kappa)} & {f2(r.agree)} & {f2(r.yes_opus)} & {f2(r.yes_gpt5)} & {f2(r.yes_gemini)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (tex / "judges_local_agreement.tex").write_text("\n".join(L))
    # figure: one row per question and arm, kappa (circle) and pairwise agreement (square)
    plt = plots.plt
    c = by_cell.copy(); c["label"] = c.axis + ", " + c.arm.str.replace("_", " ")
    c["o"] = c.axis.map(AXES.index) * 10 + c.arm.map(ARMS.index); c = c.sort_values("o", ascending=False)
    fig, ax = plt.subplots(figsize=(6.6, 0.2 * len(c) + 0.9)); y = np.arange(len(c))
    k = c.kappa.values.astype(float)
    ax.hlines(y, 0, np.nan_to_num(k), color=plots.GRID, lw=1)
    ax.plot(k, y, "o", ms=3.8, color="#2a78d6"); ax.plot(c.agree, y, "s", ms=3.2, color="#eb6834")
    ax.set_yticks(y); ax.set_yticklabels(c.label, fontsize=6.8)
    ax.set_xlabel("Fleiss kappa (circles) and pairwise agreement (squares) across the three judges"); ax.set_xlim(-0.2, 1.02)
    ax.legend(handles=[plt.Line2D([], [], marker="o", color="#2a78d6", ls=""), plt.Line2D([], [], marker="s", color="#eb6834", ls="")],
              labels=["Fleiss kappa", "pairwise agreement"], loc="lower left", fontsize=7)
    fig.tight_layout(); fig.savefig(img / "judges-local-agreement.png"); fig.savefig(img / "judges-local-agreement.pdf"); plt.close(fig)
    print(by_axis.round(2).to_string(index=False)); print("cells:", len(by_cell), "-> ", tex, img)

if __name__ == "__main__":
    main()
