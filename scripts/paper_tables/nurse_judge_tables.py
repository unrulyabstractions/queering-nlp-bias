"""Paper tables for the nurse (local bias) study that depend on the judge set.

Reads out/nurse_sub/<judge>/{scoring,greedy,estimation}.json for every judge in --judges and the merged ensemble in
out/nurse_sub/<ensemble>/{scoring,greedy,estimation}.json, and writes into --out (the paper's appendix/generated/):

  nurse/case_study_cores.tex   tab:case-study-cores   ensemble default behavior and expected deviance per arm
  nurse/judge_calibration.tex  tab:judge-agreement    per-judge vs ensemble default behavior per arm
  nurse/greedy_root.tex        tab:greedy-root        each judge's reading of the greedy root story
  nurse/judge_splits.tex       tab:judge-splits       where the judges disagree
  stats/default_statistics.tex tab:default-statistics mean vs median, mode and greedy per arm
  xeno/xeno_reweight.tex       tab:xeno-reweight      reorienting by reweighting the sampled trajectories
  nurse/values.json            every number the hand-drawn figures and the prose quote, for checking

Usage: uv run python scripts/paper_tables/nurse_judge_tables.py --judges opus gpt5 gemini haiku55 qwen deepseek \
           --ensemble all --out ~/work/papers/homo/appendix/generated
"""
from __future__ import annotations
import argparse, datetime, json, math, re
from pathlib import Path
import numpy as np
from nurse_samples_lib import hu

ROOT = Path(__file__).resolve().parents[2] / "out" / "nurse_sub"
LABEL = {"opus": "Opus 4.6", "gpt5": "GPT-5", "gemini": "Gemini 2.5 Flash",
         "haiku55": "Haiku 5.5", "qwen": "Qwen 3.7 Flash", "deepseek": "DeepSeek V3.2"}
ARMS = ["root", "trunk", "branch_1", "branch_2", "branch_3"]
AX = ["male", "female", "hetero", "same-sex"]
NUM = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

def esc(s): return s.replace("_", "\\_").replace("#", "\\#").replace("&", "\\&").replace("%", "\\%")
def load(p): return json.load(open(p))

def header(srcs, what):
    return (f"% Source: OLD/queering-nlp-bias/{srcs}\n"
            f"% Written {datetime.date.today().isoformat()} by scripts/paper_tables/nurse_judge_tables.py. {what}. Generated, do not edit by hand.\n")

def score_matrix(ens_dir):
    """n x 4 ensemble scores per arm: the 200 sampled trajectories (None read as 0, as the estimator does)."""
    sc = load(ens_dir / "scoring.json")
    return {arm: np.array([[0.0 if v is None else float(v) for v in r["method_scores"]["categorical"]] for r in sc["results"] if r["arm"] == arm]) for arm in ARMS}

def dev(B, c): return np.linalg.norm(B - c, axis=1) / math.sqrt(B.shape[1])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--judges", nargs="+", required=True); ap.add_argument("--ensemble", default="all")
    ap.add_argument("--out", required=True); a = ap.parse_args()
    out = Path(a.out).expanduser(); J = a.judges; nJ = len(J); ens = ROOT / a.ensemble
    for d in ("nurse", "stats", "xeno"): (out / d).mkdir(parents=True, exist_ok=True)
    est = load(ens / "estimation.json"); E = {arm["name"]: arm["estimates"] for arm in est["arms"]}
    S = score_matrix(ens); values = {"judges": J, "arms": {}}
    jsrc = "{" + ",".join(J) + "}"

    # ---- case-study cores
    core = {arm: S[arm].mean(0) for arm in ARMS}
    edev = {arm: dev(S[arm], core[arm]).mean() for arm in ARMS}
    for arm in ARMS:
        assert np.allclose(core[arm], E[arm]["uniform"]["core"], atol=1e-6), (arm, core[arm], E[arm]["uniform"]["core"])
        values["arms"][arm] = {"core": [round(float(x), 6) for x in core[arm]], "expected_deviance": round(float(edev[arm]), 6),
                               "deviance_var": round(float(dev(S[arm], core[arm]).var()), 6)}
    hi = {0: "malecolor", 1: "femalecolor", 2: "hetcolor", 3: "sscolor"}
    dmax, dmin = max(edev, key=edev.get), min(edev, key=edev.get); ssmax = max(ARMS, key=lambda r: core[r][3])
    L = [header(f"out/nurse_sub/{a.ensemble}/scoring.json (ensemble of {jsrc}), n = 200 sampled trajectories per arm", "Ensemble default behavior and expected deviance"),
         "\\begin{table}[hbt!]", "\\centering", "\\small",
         "\\caption{Estimated system barycenters and expected deviance by arm ($n = 200$ per arm). Deviance is dimension-normalized (\\autoref{eq:deviance}).}",
         "\\label{tab:case-study-cores}", "{\\setlength{\\fboxsep}{1pt}%", "\\def\\hival#1#2{\\fcolorbox{#1}{white}{#2}}%",
         "\\begin{tabular}{lccccc}", "\\toprule",
         "\\textbf{Arm} & $\\mathbb{E}[\\xdev]$ & $\\alpha_{\\text{male}}$ & $\\alpha_{\\text{female}}$ & $\\alpha_{\\text{hetero}}$ & $\\alpha_{\\text{same-sex}}$ \\\\", "\\midrule"]
    for arm in ARMS:
        d = hu(edev[arm]); d = f"\\textbf{{{d}}}" if arm in (dmax, dmin) else d
        cells = []
        for i, v in enumerate(core[arm]):
            c = hu(v)
            if v >= 0.5: c = f"\\hival{{{hi[i]}}}{{{c}}}"
            elif i == 3 and arm == ssmax: c = f"\\textbf{{{c}}}"
            cells.append(c)
        L.append(f"{esc(arm)} & {d} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    (out / "nurse" / "case_study_cores.tex").write_text("\n".join(L))

    # ---- per-judge calibration
    JE = {j: {arm["name"]: arm["estimates"] for arm in load(ROOT / j / "estimation.json")["arms"]} for j in J}
    L = [header(f"out/nurse_sub/{jsrc}/estimation.json and out/nurse_sub/{a.ensemble}/estimation.json, arms[*].estimates.uniform.core", "Per-judge calibration"),
         "\\begin{table}[hbt!]", "\\centering\\footnotesize",
         "\\caption{Per-judge \\emph{vs.}\\ ensemble system defaults under the uniform mean, by arm.}", "\\label{tab:judge-agreement}",
         "\\begin{tabular}{llcccc}", "\\toprule",
         "\\textbf{Arm} & \\textbf{Judge} & $\\alpha_{\\text{male}}$ & $\\alpha_{\\text{female}}$ & $\\alpha_{\\text{hetero}}$ & $\\alpha_{\\text{same-sex}}$ \\\\"]
    tag = {"branch_1": " (\\texttt{``his partner''})", "branch_2": " (\\texttt{``her partner''})", "branch_3": " (\\texttt{``their partner''})"}
    for arm in ARMS:
        L.append("\\midrule")
        for r, j in enumerate(J):
            c = JE[j][arm]["uniform"]["core"]
            first = esc(arm) if r == 0 else ("{\\scriptsize" + tag[arm] + "}" if r == 1 and arm in tag else "")
            L.append(f"{first} & {LABEL[j]} & " + " & ".join(hu(v) for v in c) + " \\\\")
            values["arms"][arm].setdefault("per_judge", {})[j] = [round(float(v), 6) for v in c]
        L.append(" & \\textit{Ensemble} & " + " & ".join(hu(v) for v in core[arm]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "nurse" / "judge_calibration.tex").write_text("\n".join(L))
    # commitment: mean of min(v, 1 - v) over every arm's uniform core (lower = more committed)
    values["commitment"] = {j: round(float(np.mean([min(v, 1 - v) for arm in ARMS for v in JE[j][arm]["uniform"]["core"]])), 4) for j in J}

    # ---- greedy root readings
    def reading(s):
        if s is None or any(v is None for v in s): return "unparseable"
        m, f, h, ss = (int(round(v)) for v in s)
        nurse = "female nurse" if f else ("male nurse" if m else "nurse gender unread")
        couple = "hetero couple" if h else ("same-sex couple" if ss else "no couple")
        return f"{nurse}, {couple}"
    L = [header(f"out/nurse_sub/{jsrc}/greedy.json, arms[0] (root).structure_scores", "Greedy root readings"),
         "\\begin{table}[hbt!]", "\\centering\\small",
         "\\caption{Greedy \\texttt{root} trajectory: each judge's $0/1$ reading of the single highest-probability story.}", "\\label{tab:greedy-root}",
         "\\begin{tabular}{l@{\\hspace{1.4em}}cccc@{\\hspace{1.6em}}l}", "\\toprule",
         "\\textbf{Judge} & $\\alpha_{\\text{male}}$ & $\\alpha_{\\text{female}}$ & $\\alpha_{\\text{hetero}}$ & $\\alpha_{\\text{same-sex}}$ & \\textbf{Reading} \\\\", "\\midrule"]
    values["greedy_root"] = {}
    for j in J:
        s = next(x for x in load(ROOT / j / "greedy.json")["arms"] if x["name"] == "root")["structure_scores"]
        values["greedy_root"][j] = s
        L.append(f"{LABEL[j]} & " + " & ".join("--" if v is None else f"${int(round(v))}$" for v in s) + f" & {reading(s)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "nurse" / "greedy_root.tex").write_text("\n".join(L))

    # ---- judge splits
    V = {j: load(ROOT / j / "scoring.json")["results"] for j in J}
    ens_sc = load(ens / "scoring.json")["results"]
    n = len(ens_sc)
    split = {arm: [0] * 4 for arm in ARMS}; minority = {j: [0] * 4 for j in J}; split_cells = []
    for i in range(n):
        arm = ens_sc[i]["arm"]
        for q in range(4):
            vs = [V[j][i]["method_scores"]["categorical"][q] for j in J]
            got = [v for v in vs if v is not None]
            if len(set(got)) > 1:
                split[arm][q] += 1; split_cells.append((i, q))
            for jj, j in enumerate(J):
                if vs[jj] is None: continue
                others = [v for k, v in enumerate(vs) if k != jj and v is not None]
                yes = sum(others); no = len(others) - yes
                if yes != no and vs[jj] != (1 if yes > no else 0): minority[j][q] += 1
    values["split_cells"] = {arm: split[arm] for arm in ARMS}; values["minority"] = minority
    order = "\\,".join(LABEL[j].split()[0] for j in J)
    L = [header(f"out/nurse_sub/{jsrc}/scoring.json (per-judge 0/1 verdicts) and out/nurse_sub/{a.ensemble}/scoring.json (ensemble mean per cell); "
                "200 sampled trajectories per arm, 4 axes. A split cell is one where the judges are not unanimous", "Judge splits"),
         "\\begin{table}[hbt!]", "\\centering\\footnotesize",
         f"\\caption{{Where the {NUM[nJ]} judges disagree. Top: number of split cells (the judges are not unanimous) per arm and axis, out of 200 sampled trajectories each. "
         f"Middle: for each axis, how often each judge votes against the majority of the other {NUM[nJ - 1]}. Bottom: five split cells with the {NUM[nJ]} verdicts "
         f"({', '.join(LABEL[j] for j in J)}) and the ensemble score.}}",
         "\\label{tab:judge-splits}", "\\begin{tabular}{lrrrr}", "\\toprule",
         "\\textbf{Arm} & \\textbf{male} & \\textbf{female} & \\textbf{hetero} & \\textbf{same-sex} \\\\", "\\midrule"]
    for arm in ARMS: L.append(f"{esc(arm)} & " + " & ".join(str(x) for x in split[arm]) + " \\\\")
    L += ["\\midrule", "\\textbf{against the majority} & & & & \\\\"]
    for j in J: L.append(f"\\quad {LABEL[j]} & " + " & ".join(str(x) for x in minority[j]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\par\\medskip",
          "\\begin{tabular}{@{}llc>{\\raggedright\\arraybackslash}p{0.5\\linewidth}@{}}", "\\toprule",
          "\\textbf{Cell} & \\textbf{Axis} & \\textbf{Verdicts} & \\textbf{Trajectory (opening)} \\\\", "\\midrule"]
    wanted = [(1, 1), (403, 1), (400, 3), (410, 3), (305, 2)]
    pick = [c for c in wanted if c in set(split_cells)]
    for c in split_cells:
        if len(pick) >= 5: break
        if c not in pick and all(c[1] != p[1] or ens_sc[c[0]]["arm"] != ens_sc[p[0]]["arm"] for p in pick): pick.append(c)
    for i, q in pick[:5]:
        vs = [V[j][i]["method_scores"]["categorical"][q] for j in J]
        e = ens_sc[i]["method_scores"]["categorical"][q]
        txt = " ".join(ens_sc[i]["text"].split())[:230].rsplit(" ", 1)[0]
        L.append(f"{esc(ens_sc[i]['arm'])} \\#{i} & {AX[q]} & " + "\\,".join("--" if v is None else str(v) for v in vs)
                 + f" $\\to$ {hu(e, 2)} & {{\\scriptsize\\itshape {esc(txt)}\\ldots}} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "nurse" / "judge_splits.tex").write_text("\n".join(L))

    # ---- default statistics
    L = [header(f"out/nurse_sub/{a.ensemble}/estimation.json, arms[*].estimates.{{uniform,median,mode,greedy}}.core", "Default behavior under four statistics"),
         "\\begin{table}[hbt!]", "\\centering\\footnotesize",
         "\\caption{Default behavior under four statistics on every arm of the nurse study. For each arm, the largest per-axis gap between the mean and the median, the mode, and the single greedy decode, and whether the axes at or above one half are the same set under the mean, the median and the mode (agree) and under the greedy decode (greedy).}",
         "\\label{tab:default-statistics}", "\\begin{tabular}{lrrrcc}", "\\toprule",
         "\\textbf{Arm} & $\\max|\\text{mean}-\\text{median}|$ & $\\max|\\text{mean}-\\text{mode}|$ & $\\max|\\text{mean}-\\text{greedy}|$ & \\textbf{agree} & \\textbf{greedy} \\\\", "\\midrule"]
    values["statistics"] = {}
    for arm in ARMS:
        u, md, mo, g = (np.array(E[arm][k]["core"]) for k in ("uniform", "median", "mode", "greedy"))
        top = lambda v: tuple(np.where(v >= 0.5)[0])
        agree = top(u) == top(md) == top(mo); gre = top(u) == top(g)
        row = [np.abs(u - md).max(), np.abs(u - mo).max(), np.abs(u - g).max()]
        values["statistics"][arm] = {"median": md.tolist(), "mode": mo.tolist(), "greedy": g.tolist(), "gaps": [round(float(x), 4) for x in row]}
        L.append(f"{esc(arm)} & " + " & ".join(hu(x, 2) for x in row) + f" & {'yes' if agree else 'no'} & {'yes' if gre else 'no'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "stats" / "default_statistics.tex").write_text("\n".join(L))

    # ---- xeno reweight
    betas = [0, 2, 4, 8, 16]
    L = [header(f"out/nurse_sub/{a.ensemble}/scoring.json (ensemble judge scores, n = 200 sampled trajectories per arm)", "Reorienting by reweighting"),
         "% Method: P_beta(y|x_p) propto P_0(y|x_p) exp(beta * dev(y|x_p)), P_0 uniform over the n sampled trajectories, so every reoriented",
         "%   quantity is the self-normalized weighted mean with w_k = exp(beta dev_k) / sum_j exp(beta dev_j), dev_k = ||B(y_k) - <B>(x_p)||_2 / sqrt(dim B).",
         "%   zbar_i = weighted mean score on axis i; E[d], Var[d] = normalized deviance from the row's own reoriented default behavior, under P_beta;",
         "%   E[r] = weighted mean of dev_k (the stray reward); H, e^H = entropy of the normalized reoriented default behavior (nats); ESS/n = 1 / (n sum_k w_k^2).",
         "\\providecommand{\\zbar}{\\bar{z}}", "\\begin{table*}[hbt!]", "\\centering\\small",
         "\\caption{Reorienting the nurse distribution by reweighting the sampled\ntrajectories ($n = 200$ per arm, no new generations). Each row is the\nreoriented distribution $P_{\\beta} \\propto P_0 e^{\\beta \\xdev}$ read off the\nsame sample. $\\beta = 0$ is the unintervened estimate and reproduces\n\\autoref{tab:case-study-cores}. $\\mathbb{E}[\\xdev]$ and\n$\\operatorname{Var}[\\xdev]$ are measured from each row's own reoriented default\nbehavior; $\\mathbb{E}[\\xenoreward_{\\chi}]$ is the stray reward, measured from\nthe unintervened default behavior. Deviance and the stray reward are on the\ndimension-normalized scale of \\autoref{eq:deviance}. ESS$/n$ is the fraction of the sample the\nreweighted estimate uses.}",
         "\\label{tab:xeno-reweight}", "\\fitcol{%", "\\begin{tabular}{ll rrrr rrr rr r}", "\\toprule",
         "\\textbf{Arm} & $\\beta$ & $\\zbar_{\\text{male}}$ & $\\zbar_{\\text{female}}$ & $\\zbar_{\\text{hetero}}$ & $\\zbar_{\\text{same-sex}}$ & $\\mathbb{E}[\\xdev]$ & $\\operatorname{Var}[\\xdev]$ & $\\mathbb{E}[\\xenoreward_{\\chi}]$ & $H$ & $e^{H}$ & ESS$/n$ \\\\"]
    values["xeno"] = {}
    for arm in ARMS:
        B = S[arm]; nn = len(B); r = dev(B, B.mean(0)); L.append("\\midrule")
        for bi, b in enumerate(betas):
            w = np.exp(b * (r - r.max())); w /= w.sum()
            z = w @ B; d = dev(B, z); Ed = w @ d; Vd = w @ (d - Ed) ** 2; Er = w @ r
            p = z / z.sum(); H = -sum(x * math.log(x) for x in p if x > 0); ess = 1 / (nn * (w ** 2).sum())
            values["xeno"].setdefault(arm, {})[b] = [round(float(x), 6) for x in (*z, Ed, Vd, Er, H, math.exp(H), ess)]
            L.append((f"{esc(arm)}" if bi == 0 else "") + f" & {b} & " + " & ".join(hu(x) for x in (*z, Ed, Vd, Er, H, math.exp(H), ess)) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    (out / "xeno" / "xeno_reweight.tex").write_text("\n".join(L))

    # ---- greedy generations, verbatim (Appendix C.6)
    gr = load(ens / "greedy.json")
    def tex_text(t):
        t = t.replace("\\", "\\textbackslash{}")
        for a, b in [("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")]:
            t = t.replace(a, b)
        t = re.sub(r'"([^"\n]*)"', r"``\1''", t)  # TeX quotes for the story's straight double quotes
        return "\n\n".join(" ".join(par.split()) for par in t.strip().split("\n\n"))
    head = {"root": "\\textbf{Root} (no continuation prefix):", "trunk": "\\textbf{Trunk} (\\texttt{``At the altar, the nurse and''}):",
            "branch_1": "\\textbf{Branch\\_1} (\\texttt{``his partner''}):", "branch_2": "\\textbf{Branch\\_2} (\\texttt{``her partner''}):",
            "branch_3": "\\textbf{Branch\\_3} (\\texttt{``their partner''}):"}
    L = [header(f"out/nurse_sub/{a.ensemble}/greedy.json (Claude Haiku 4.5, temperature 0, up to 512 new tokens, one decode per arm)", "Greedy generations, verbatim")]
    for arm in ARMS:
        g = next(x for x in gr["arms"] if x["name"] == arm)
        L += [head[arm], "\\begin{quote}", "\\small\\itshape " + tex_text(g["text"]), "\\end{quote}", ""]
    (out / "nurse" / "greedy_generations.tex").write_text("\n".join(L))

    json.dump(values, open(out / "nurse" / "values.json", "w"), indent=1)
    print("nurse judge tables ->", out)

if __name__ == "__main__":
    main()
