"""Emit the LaTeX tables for the global-bias appendix from steering/out/<run>/<ANALYSIS_DIR>/.

Usage: STEERING_ARMS=F,EL,IL uv run python steering/tables_tex.py --run full --out <paper>/appendix/generated
Writes global-bias/{configuration,prompts,comparison-formal,comparison-explicit-implicit,axes-catalogue}.tex and
judges/global-agreement.tex. Every file starts with a source comment; nothing here is hand-typed from memory.
"""
from __future__ import annotations
import argparse, json, datetime, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from axes import AXES, JUDGE_AXES
from select_axes import moving_axes, comparisons, largest, ORDER5
from prompts import ARMS, ORDER, ANALYSIS_DIR

HERE = Path(__file__).resolve().parent
ARM_DESC = {"F": "formal, no identity", "D": "formal, ``I am a Democrat''", "EL": "formal, ``I am Mexican-American''",
            "IL": "casual, light Spanglish, San Antonio", "HL": "casual, Spanglish, ``I'm Mexican-American''",
            "EP": "formal, ``I am Polish-American''", "IP": "casual, babcia and pierogi, Buffalo", "HP": "casual, babcia, ``I'm Polish-American''"}
def tex_escape(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("$", "\\$").replace("#", "\\#")
             .replace("_", "\\_").replace("{", "\\{").replace("}", "\\}").replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}"))

def header(run, what):
    return (f"% Source: steering/out/{run}/{ANALYSIS_DIR}/ in the pipeline repo (summary.csv, contrasts.csv, agreement.csv),\n"
            f"%         written by steering/tables_tex.py on {datetime.date.today().isoformat()}. {what}. Generated, do not edit by hand.\n")

CMP_FILE = {"lat_f": "comparison-formal.tex", "lat_ei": "comparison-explicit-implicit.tex"}
COMP_HEAD = {"lat_pol": "Lat$-$Pol", "lat_f": "Lat$-$F", "pol_f": "Pol$-$F", "lat_ei": "Lat E$-$I", "pol_ei": "Pol E$-$I"}
from prompts import SUBSET
# Only the formal and Mexican-American arms are reported: drop Polish-only examples and axes from the display text.
POLISH_ONLY = {"polish_heritage_share", "enclave_offered_polish_list"} if (SUBSET and not ({"EP", "IP", "HP"} & set(ORDER))) else set()
if POLISH_ONLY: COMP_HEAD.update({"lat_f": "MA$-$F", "lat_ei": "E$-$I"})
NCOMP = {2: "two", 3: "three", 4: "four", 5: "five"}[len(ORDER5)]
NARMS = {3: "three", 8: "eight"}.get(len(ORDER), str(len(ORDER)))
def scrub(q):
    if not POLISH_ONLY: return q
    for a, b in [("Spanish (Latino arms) or Polish (Polish arms)", "Spanish"), ("Spanish or Polish", "Spanish"),
                 (", Polish, Democrat", ""), ("as a Democrat, ", ""), (", Buffalo", ""), (", St. Adalbert, Polish delis, pierogi, a Polish parish, St. Hyacinth, Copernicus Center", ""),
                 ("Latino or Slavic group", "Latino group"), ("; a Ukrainian area to a Polish-American", ""),
                 (", Polish, Polish-American, Polonia, Eastern European, Slavic", ""), (", Little Warsaw, Polish capital", ""),
                 (", czesc, powodzenia, pozdrawiam, sto lat, na zdrowie", ""), ("; na zdrowie as a joke, dobrze as cool, kielbasa jokes", ""),
                 (", Polish diacritics and case endings", ""), ("what pierogi are, ", ""), (", Polish Catholic traditions", ""), ("; babcia, pierogi)", ")"), ("; kochana, kochanie, przyjacielu;", ";"), ("('babcia (grandmother)', 'your mama, your mom')", "('your mama, your mom')"), (" ('your grandmother' where the user wrote 'babcia')", " ('your mom' where the user wrote 'mama')")]:
        q = q.replace(a, b)
    return q

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    an = HERE / "out" / args.run / ANALYSIS_DIR; out = Path(args.out)
    global_dir, judges_dir = out / "global-bias", out / "judges"
    global_dir.mkdir(parents=True, exist_ok=True); judges_dir.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(an / "summary.csv"); agree = pd.read_csv(an / "agreement.csv")
    from plots import all_rows
    LAB = {k: lab for k, lab, _ in all_rows(summ)}
    lab = lambda k: LAB.get(k, k.replace("_", " "))
    resp = [json.loads(l) for l in (HERE / "out" / args.run / "responses.jsonl").open() if l.strip()]
    resp = [r for r in resp if r["arm"] in ORDER]
    n_per_arm = pd.Series([r["arm"] for r in resp]).value_counts().min()
    # config
    cfg = [header(args.run, "Experiment configuration"), "\\begin{table}[hbt!]", "\\centering\\small", "\\caption{Steering experiment configuration at a glance.}",
           "\\label{tab:steering-config}", "\\begin{tabular}{@{}l>{\\raggedright\\arraybackslash}p{0.8\\linewidth}@{}}", "\\toprule",
           f"Target       & Gemini 2.5 Flash-Lite (\\texttt{{{tex_escape(resp[0]['model'])}}}), served by Google AI Studio through OpenRouter \\\\",
           f"Decoding     & temperature $\\tau = {resp[0]['temperature']:.1f}$, up to ${resp[0]['max_tokens']}$ new tokens, single turn, no system prompt; ${sum(1 for r in resp if r.get('finish') == 'length')}$ of ${len(resp)}$ responses reached the cap and were kept \\\\",
           f"Arms         & {len(ORDER)} (\\autoref{{tab:steering-prompts}}); ${n_per_arm}$ samples per arm, ${len(resp)}$ responses \\\\",
           "Judges       & Claude Haiku 5.5 (\\texttt{claude-haiku-5-5}), Qwen 3.7 Flash (\\texttt{qwen/qwen3.7-flash}, reasoning disabled), DeepSeek V3.2 (\\texttt{deepseek/deepseek-v3.2}, reasoning disabled); each returns a JSON object constrained to the batch schema \\\\",
           "Judge protocol & eight batched calls per response, grouped by axis family, the rubric beside each question; evidence quote then yes/no, JSON schema output \\\\",
           "Blinding     & identity, constraint and preference batches see the formal request in place of the user's message \\\\",
           "Judge-free axes & gazetteer crosswalk to the 77 community areas, one census vintage, CTA rail table, pinned lexicons \\\\",
           "Estimator    & per-arm mean of the three-judge ensemble score; bootstrap and Wilson 95\\% intervals; " + ("two group comparisons (Mexican-American vs formal, explicit vs implicit)" if POLISH_ONLY else "five group comparisons (Latino vs Polish, each group vs formal, explicit vs implicit within each group)") + ", permutation tests with Holm correction \\\\",
           "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (global_dir / "configuration.tex").write_text("\n".join(cfg))
    # prompts
    pr = [header(args.run, "The arms verbatim"), "\\begin{table}[hbt!]", "\\centering\\scriptsize",
          f"\\caption{{The {NARMS} arms. One request, the identity signal varied; every arm states the same budget, commute, transit and quiet.}}",
          "\\label{tab:steering-prompts}", "\\begin{tabular}{@{}l>{\\raggedright\\arraybackslash}p{0.17\\linewidth}>{\\raggedright\\arraybackslash}p{0.72\\linewidth}@{}}",
          "\\toprule", "\\textbf{arm} & \\textbf{signal} & \\textbf{user message} \\\\", "\\midrule"]
    for a in ORDER:
        pr.append(f"{a} & {ARM_DESC[a]} & \\textit{{{tex_escape(ARMS[a]['text'])}}} \\\\[2pt]")
    pr += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (global_dir / "prompts.tex").write_text("\n".join(pr))
    # every axis, grouped by family, by largest gap within the family
    fam = {a["id"]: a["family"] for a in AXES}
    piv_all = summ.pivot(index="axis", columns="arm", values="mean")
    ident = [c for c in piv_all.columns if c not in ("F", "D")]
    gap = pd.concat([(piv_all[ident].max(axis=1) - piv_all["F"]).abs(), (piv_all[ident].min(axis=1) - piv_all["F"]).abs()], axis=1).max(axis=1)
    fams = ("steering", "identity", "constraints", "preferences", "service", "form")
    def ordered(ids):
        out_ = []
        for f in fams:
            ks = sorted([k for k in ids if fam.get(k) == f], key=lambda k: -float(gap.get(k, 0) or 0))
            if ks: out_ += ["#" + f] + ks
        return out_
    comp = comparisons(an); keep = moving_axes(comp, analysis_dir=an)
    # one table per comparison: axes whose difference in that comparison is at least 0.2, by difference
    kind_of = {a["id"]: ("J" if a["kind"] == "judge" else "M") for a in AXES}
    GROUP_NAMES = {"lat_pol": ("Latino", "Polish"), "lat_f": ("Latino", "formal"), "pol_f": ("Polish", "formal"),
                   "lat_ei": ("explicit", "implicit"), "pol_ei": ("explicit", "implicit")}
    CAPTION = {"lat_pol": "the three Mexican-American arms pooled and the three Polish-American arms pooled",
               "lat_f": "the three Mexican-American arms pooled and the formal baseline",
               "pol_f": "the three Polish-American arms pooled and the formal baseline",
               "lat_ei": "the explicit and the implicit Mexican-American arms",
               "pol_ei": "the explicit and the implicit Polish-American arms"}
    if POLISH_ONLY:
        GROUP_NAMES["lat_f"] = ("Mexican-American", "formal")
        CAPTION["lat_f"] = "the two Mexican-American arms pooled and the formal baseline"
    for k in [x for x in ORDER5 if x in set(comp.comparison)]:
        g = comp[(comp.comparison == k) & (comp["diff"].abs() >= 0.2) & comp.axis.isin(keep)].sort_values("diff", ascending=False)
        a_, b_ = GROUP_NAMES[k]
        tl = [header(args.run, f"Comparison {k}"), "\\begin{table}[h!]", "\\centering\\footnotesize",
              f"\\caption{{Axes whose mean score differs by at least 0.2 between {CAPTION[k]}. Kind: J = judge, M = measured; judge axes with Fleiss' $\\kappa < 0.5$ are left out. Every score is in $[0,1]$. $^{{*}}$: permutation test, Holm-adjusted across axes within the comparison, $p<0.05$.}}",
              f"\\label{{tab:steering-cmp-{k.replace('_','-')}}}", "\\begin{tabular}{llrrr}", "\\toprule",
              f"\\textbf{{axis}} & \\textbf{{kind}} & \\textbf{{{a_}}} & \\textbf{{{b_}}} & \\textbf{{difference}} \\\\", "\\midrule"]
        for _, r in g.iterrows():
            star = "$^{*}$" if r.p_holm < 0.05 else ""
            dv = f"{r['diff']:+.2f}".replace("+", "$+$").replace("-", "$-$")
            tl.append(f"{tex_escape(lab(r.axis))} & {kind_of.get(r.axis, 'M')} & {r.mean_a:.2f} & {r.mean_b:.2f} & {dv}{star} \\\\")
        tl += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
        (global_dir / CMP_FILE.get(k, f"comparison-{k}.tex")).write_text("\n".join(tl))
    # agreement over every judge axis
    ag = agree[~agree.axis.str.contains("@")].set_index("axis")
    all_judge = [a["id"] for a in JUDGE_AXES if a["id"] in ag.index]
    ag_ids = [k for k in ordered(all_judge) if k.startswith("#") or k in ag.index]
    al = [header(args.run, "Judge agreement"), "\\begin{center}\\footnotesize", "\\begin{longtable}{@{}lrrrrr@{}}",
          f"\\caption{{Judge agreement on every judge axis: Fleiss' $\\kappa$ and pairwise agreement across the three judges, and each judge's yes rate pooled over arms. Axes with $\\kappa < 0.5$ (marked $^{{\\dagger}}$) are not used in the results. Mean $\\kappa$: {ag.loc[all_judge].fleiss_kappa.mean():.2f}; mean pairwise agreement: {ag.loc[all_judge].pairwise_agreement.mean():.2f}.}}\\label{{tab:steering-agreement}}\\\\",
          "\\toprule \\textbf{axis} & $\\kappa$ & \\textbf{agree} & \\textbf{Haiku 5.5} & \\textbf{Qwen 3.7 Flash} & \\textbf{DeepSeek V3.2} \\\\ \\midrule \\endfirsthead",
          "\\toprule \\textbf{axis} & $\\kappa$ & \\textbf{agree} & \\textbf{Haiku 5.5} & \\textbf{Qwen 3.7 Flash} & \\textbf{DeepSeek V3.2} \\\\ \\midrule \\endhead",
          "\\bottomrule \\endfoot"]
    for ax in ag_ids:
        if ax.startswith("#"):
            al.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textbf{{{tex_escape(ax[1:])}}}}} \\\\"); continue
        row = ag.loc[ax]
        f2 = lambda v: "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.2f}"
        dag = "$^{\\dagger}$" if (isinstance(row.fleiss_kappa, float) and (np.isnan(row.fleiss_kappa) or row.fleiss_kappa < 0.5)) else ""
        al.append(f"{tex_escape(lab(ax))}{dag} & {f2(row.fleiss_kappa)} & {f2(row.pairwise_agreement)} & {f2(row.get('yes_haiku55', float('nan')))} & {f2(row.get('yes_qwen', float('nan')))} & {f2(row.get('yes_deepseek', float('nan')))} \\\\")
    al += ["\\end{longtable}", "\\end{center}", ""]
    (judges_dir / "global-agreement.tex").write_text("\n".join(al))
    # catalogue, with the largest difference between two arms and which arms
    def maxdiff(ax):
        d, k = largest(comp, ax)
        if k is None: return "--", "--"
        return f"{d:.2f}", COMP_HEAD[k]
    cl = [header(args.run, "The candidate axes"), "\\begin{center}\\scriptsize",
          "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{0.19\\linewidth}l>{\\raggedright\\arraybackslash}p{0.55\\linewidth}rl@{}}",
          f"\\caption{{The {len([a for a in AXES if a['id'] not in POLISH_ONLY])} behavior axes. Kind: J = yes/no answered by the judge ensemble, M = measured from the text and the pinned tables. Every axis is a score in $[0,1]$; counts are divided by their largest value over all responses. $\\Delta$: the largest absolute difference among the {NCOMP} group comparisons, and which comparison: " + ("MA = the two Mexican-American arms pooled, F = formal, E$-$I = explicit against implicit." if POLISH_ONLY else "Lat = the three Mexican-American arms pooled, Pol = the three Polish-American arms pooled, F = formal, E$-$I = explicit against implicit within one group.") + f" The judge rubrics, lexicons and tables are in the pipeline repository.}}\\label{{tab:steering-axes}}\\\\",
          "\\toprule \\textbf{axis} & \\textbf{kind} & \\textbf{question} & $\\Delta$ & \\textbf{comparison} \\\\ \\midrule \\endfirsthead",
          "\\toprule \\textbf{axis} & \\textbf{kind} & \\textbf{question} & $\\Delta$ & \\textbf{comparison} \\\\ \\midrule \\endhead", "\\bottomrule \\endfoot"]
    for fam in ("steering", "identity", "constraints", "preferences", "service", "form"):
        cl.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{{tex_escape(fam)}}}}} \\\\")
        for a in [x for x in AXES if x["family"] == fam and x["id"] not in POLISH_ONLY]:
            d, arms = maxdiff(a["id"])
            cl.append(f"{tex_escape(lab(a['id']))} & {'J' if a['kind']=='judge' else 'M'} & {tex_escape(scrub(a['q']))} & {d} & {arms} \\\\")
    cl += ["\\end{longtable}", "\\end{center}", ""]
    (global_dir / "axes-catalogue.tex").write_text("\n".join(cl))
    print("tables ->", out)

if __name__ == "__main__":
    main()
