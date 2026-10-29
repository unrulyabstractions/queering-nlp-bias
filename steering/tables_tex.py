"""Emit the LaTeX tables for the steering appendix from steering/out/<run>/analysis/.

Usage: uv run python steering/tables_tex.py --run full --out /Users/unrulyabstractions/work/papers/homo/appendix/generated/steering [--top 16]
Files: steering_config.tex, steering_prompts.tex, steering_rates.tex, steering_service.tex, steering_agreement.tex,
       steering_axes_catalogue.tex. Every file starts with a source comment; nothing here is hand-typed from memory.
"""
from __future__ import annotations
import argparse, json, datetime, sys, re
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from axes import AXES, JUDGE_AXES
from prompts import ARMS, ORDER

HERE = Path(__file__).resolve().parent
ARM_DESC = {"F": "formal, no identity", "D": "formal, ``I am a Democrat''", "EL": "formal, ``I am Mexican-American''",
            "IL": "casual, light Spanglish, San Antonio", "HL": "casual, Spanglish, ``I'm Mexican-American''",
            "EP": "formal, ``I am Polish-American''", "IP": "casual, babcia and pierogi, Buffalo", "HP": "casual, babcia, ``I'm Polish-American''"}
HEADLINE_SCRIPT = ["enclave_offered_latino_list", "enclave_offered_polish_list", "hispanic_share_mean", "white_nh_share_mean",
                   "black_nh_share_mean", "n_neighborhoods", "offrail_share", "commute_fail_share", "l2_tokens_count",
                   "l2_greeting_signoff", "political_descriptors_count", "identity_terms_count", "identity_in_opening", "word_count",
                   "rent_figures_count", "exclamation_density"]

def tex_escape(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("$", "\\$").replace("#", "\\#")
             .replace("_", "\\_").replace("{", "\\{").replace("}", "\\}").replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}"))

def header(run, what):
    return (f"% Source: steering/out/{run}/analysis/ in the pipeline repo (summary.csv, contrasts.csv, agreement.csv),\n"
            f"%         written by steering/tables_tex.py on {datetime.date.today().isoformat()}. {what}. Generated, do not edit by hand.\n")

def fmt(v, kind):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "--"
    if kind == "rate": return f"{v:.2f}"
    if kind == "pct": return f"{v:.0f}"
    if kind == "int": return f"{v:.1f}"
    return f"{v:.2f}"

def rates_table(summ, contr, axes_ids, label, caption, kinds, run):
    piv = summ.pivot(index="axis", columns="arm", values="mean").reindex(axes_ids)[ORDER]
    holm = contr.pivot(index="axis", columns="arm", values="p_holm").reindex(axes_ids) if len(contr) else None
    lines = [header(run, caption.split(".")[0]), "\\begin{table}[hbt!]", "\\centering\\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{l" + "r" * len(ORDER) + "}", "\\toprule",
             "\\textbf{axis} & " + " & ".join(f"\\textbf{{{a}}}" for a in ORDER) + " \\\\", "\\midrule"]
    for ax in axes_ids:
        if ax not in piv.index: continue
        cells = []
        for arm in ORDER:
            v = piv.loc[ax, arm]; s = fmt(v, kinds.get(ax, "rate"))
            if holm is not None and arm != "F" and ax in holm.index:
                p = holm.loc[ax, arm]
                if isinstance(p, float) and not np.isnan(p) and p < 0.05 and s != "--": s += "$^{*}$"
            cells.append(s)
        lines.append(tex_escape(ax.replace("_", " ")) + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\par\\smallskip", "\\footnotesize $^{*}$ differs from the formal baseline, permutation test, Holm-adjusted within the arm, $p<0.05$.",
              "\\end{table}", ""]
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True); ap.add_argument("--top", type=int, default=16)
    args = ap.parse_args()
    an = HERE / "out" / args.run / "analysis"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(an / "summary.csv"); contr = pd.read_csv(an / "contrasts.csv"); agree = pd.read_csv(an / "agreement.csv")
    meta = json.load((an / "summary.json").open())
    resp = [json.loads(l) for l in (HERE / "out" / args.run / "responses.jsonl").open() if l.strip()]
    n_per_arm = pd.Series([r["arm"] for r in resp]).value_counts().min()
    # config
    cfg = [header(args.run, "Experiment configuration"), "\\begin{table}[hbt!]", "\\centering\\small", "\\caption{Steering experiment configuration at a glance.}",
           "\\label{tab:steering-config}", "\\begin{tabular}{@{}l>{\\raggedright\\arraybackslash}p{0.8\\linewidth}@{}}", "\\toprule",
           f"Target       & Gemini 2.5 Flash-Lite (\\texttt{{{tex_escape(resp[0]['model'])}}}), served by Google AI Studio through OpenRouter \\\\",
           f"Decoding     & temperature $\\tau = {resp[0]['temperature']:.1f}$, up to ${resp[0]['max_tokens']}$ new tokens, single turn, no system prompt; ${sum(1 for r in resp if r.get('finish') == 'length')}$ of ${len(resp)}$ responses reached the cap and were kept \\\\",
           f"Arms         & {len(ORDER)} (\\autoref{{tab:steering-prompts}}); ${n_per_arm}$ samples per arm, ${len(resp)}$ responses \\\\",
           "Judges       & Claude Haiku 5.5 (\\texttt{claude-haiku-5-5}), GPT-5-nano (\\texttt{gpt-5-nano-2025-08-07}, reasoning effort low), Qwen 3.7 Flash (\\texttt{qwen/qwen3.7-flash}, reasoning disabled); each returns a JSON object constrained to the batch schema \\\\",
           "Judge protocol & six batched calls per response: four in the first round (identity, constraints, preferences, form) and two in the second (the rewritten axes, blind and open); evidence quote then yes/no, JSON schema output \\\\",
           "Blinding     & identity, constraint and preference batches see the formal request in place of the user's message \\\\",
           "Judge-free axes & gazetteer crosswalk to the 77 community areas, one census vintage, CTA rail table, pinned lexicons \\\\",
           "Estimator    & per-arm mean of the three-judge ensemble score; bootstrap and Wilson 95\\% intervals; permutation tests against the formal arm with Holm correction \\\\",
           "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "steering_config.tex").write_text("\n".join(cfg))
    # prompts
    pr = [header(args.run, "The eight arms verbatim"), "\\begin{table}[hbt!]", "\\centering\\scriptsize",
          "\\caption{The eight arms. One request, the identity signal varied; every arm states the same budget, commute, transit and quiet.}",
          "\\label{tab:steering-prompts}", "\\begin{tabular}{@{}l>{\\raggedright\\arraybackslash}p{0.17\\linewidth}>{\\raggedright\\arraybackslash}p{0.72\\linewidth}@{}}",
          "\\toprule", "\\textbf{arm} & \\textbf{signal} & \\textbf{user message} \\\\", "\\midrule"]
    for a in ORDER:
        pr.append(f"{a} & {ARM_DESC[a]} & \\textit{{{tex_escape(ARMS[a]['text'])}}} \\\\[2pt]")
    pr += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "steering_prompts.tex").write_text("\n".join(pr))
    # headline judge axes by gap
    judge_ids = [a["id"] for a in JUDGE_AXES if a["id"] in set(summ.axis)]
    piv = summ[summ.axis.isin(judge_ids)].pivot(index="axis", columns="arm", values="mean")
    ident = [c for c in piv.columns if c not in ("F", "D")]
    gap = pd.concat([(piv[ident].max(axis=1) - piv["F"]).abs(), (piv[ident].min(axis=1) - piv["F"]).abs()], axis=1).max(axis=1)
    top = gap.sort_values(ascending=False).index.tolist()[: args.top]
    kinds = {}
    (out / "steering_rates.tex").write_text(rates_table(summ, contr, top, "tab:steering-rates",
        f"Judge axes with the largest gap between an identity arm and the formal baseline: ensemble yes rate per arm, three judges averaged, ${n_per_arm}$ responses per arm.", kinds, args.run))
    kinds = {"hispanic_share_mean": "pct", "white_nh_share_mean": "pct", "black_nh_share_mean": "pct", "n_neighborhoods": "int",
             "l2_tokens_count": "int", "word_count": "int", "rent_figures_count": "int", "political_descriptors_count": "int", "exclamation_density": "int",
             "identity_terms_count": "int"}
    sids = [s for s in HEADLINE_SCRIPT if s in set(summ.axis)]
    (out / "steering_service.tex").write_text(rates_table(summ, contr, sids, "tab:steering-service",
        "Judge-free axes per arm: shares of responses, mean census shares (percent) of the offered community areas, and counts.", kinds, args.run))
    # agreement
    ag = agree[~agree.axis.str.endswith("@r1")].set_index("axis").reindex(top).dropna(subset=["fleiss_kappa"])
    al = [header(args.run, "Judge agreement"), "\\begin{table}[hbt!]", "\\centering\\footnotesize",
          f"\\caption{{Judge agreement on the headline axes: Fleiss' $\\kappa$ and pairwise agreement across the three judges, and each judge's yes rate pooled over arms. Mean $\\kappa$ over all judge axes: {agree[~agree.axis.str.endswith('@r1')].fleiss_kappa.mean():.2f}; mean pairwise agreement: {agree[~agree.axis.str.endswith('@r1')].pairwise_agreement.mean():.2f}.}}",
          "\\label{tab:steering-agreement}", "\\begin{tabular}{lrrrrr}", "\\toprule",
          "\\textbf{axis} & $\\kappa$ & \\textbf{agree} & \\textbf{Haiku 5.5} & \\textbf{GPT-5-nano} & \\textbf{Qwen 3.7 Flash} \\\\", "\\midrule"]
    for ax, row in ag.iterrows():
        al.append(f"{tex_escape(ax.replace('_',' '))} & {row.fleiss_kappa:.2f} & {row.pairwise_agreement:.2f} & {row.get('yes_haiku55', float('nan')):.2f} & {row.get('yes_nano', float('nan')):.2f} & {row.get('yes_qwen', float('nan')):.2f} \\\\")
    al += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "steering_agreement.tex").write_text("\n".join(al))
    # second judging round: agreement before and after the rewrite
    r2 = [a for a in JUDGE_AXES if "rubric_round1" in a]
    agi = agree.set_index("axis")
    rl = [header(args.run, "Second judging round"), "\\begin{table}[hbt!]", "\\centering\\footnotesize",
          "\\caption{Axes re-judged with a rewritten rubric after the first round: Fleiss' $\\kappa$ and pairwise agreement among the three judges before and after, the number of responses with a first-round vote from all three judges, and the ensemble yes rate pooled over arms after.}",
          "\\label{tab:steering-refine}", "\\begin{tabular}{lrrrrrr}", "\\toprule",
          "\\textbf{axis} & $n$ before & $\\kappa$ before & agree before & $\\kappa$ after & agree after & yes rate after \\\\", "\\midrule"]
    for a in r2:
        b = agi.loc[a["id"] + "@r1"] if a["id"] + "@r1" in agi.index else None; c = agi.loc[a["id"]] if a["id"] in agi.index else None
        rl.append(f"{tex_escape(a['id'].replace('_',' '))} & " + " & ".join([
            f"{int(b.n)}" if b is not None else "--",
            f"{b.fleiss_kappa:.2f}" if b is not None else "--", f"{b.pairwise_agreement:.2f}" if b is not None else "--",
            f"{c.fleiss_kappa:.2f}" if c is not None else "--", f"{c.pairwise_agreement:.2f}" if c is not None else "--",
            f"{c.yes_rate_overall:.2f}" if c is not None else "--"]) + " \\\\")
    rl += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (out / "steering_refine.tex").write_text("\n".join(rl))
    # catalogue
    cl = [header(args.run, "The candidate axes"), "\\begin{center}\\scriptsize",
          "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{0.21\\linewidth}l>{\\raggedright\\arraybackslash}p{0.66\\linewidth}@{}}",
          f"\\caption{{The {len(AXES)} candidate behavior axes. Kind: J = yes/no answered by the judge ensemble, S = computed by script from the text and the pinned tables. A dagger marks an axis whose rubric was rewritten for the second judging round or that was added after the first round. The judge rubrics, lexicons and tables are in the pipeline repository.}}\\label{{tab:steering-axes}}\\\\",
          "\\toprule \\textbf{axis} & \\textbf{kind} & \\textbf{question} \\\\ \\midrule \\endfirsthead",
          "\\toprule \\textbf{axis} & \\textbf{kind} & \\textbf{question} \\\\ \\midrule \\endhead", "\\bottomrule \\endfoot"]
    for fam in ("steering", "identity", "constraints", "preferences", "service", "form"):
        cl.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\textbf{{{tex_escape(fam)}}}}} \\\\")
        for a in [x for x in AXES if x["family"] == fam]:
            cl.append(f"{tex_escape(a['id'].replace('_',' '))}{'$^{\\dagger}$' if a.get('round') == 2 else ''} & {'J' if a['kind']=='judge' else 'S'} & {tex_escape(a['q'])} \\\\")
    cl += ["\\end{longtable}", "\\end{center}", ""]
    (out / "steering_axes_catalogue.tex").write_text("\n".join(cl))
    print("tables ->", out, "| headline axes:", top)

if __name__ == "__main__":
    main()
