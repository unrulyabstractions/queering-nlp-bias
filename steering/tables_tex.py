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
from select_axes import moving_axes
from prompts import ARMS, ORDER

HERE = Path(__file__).resolve().parent
ARM_DESC = {"F": "formal, no identity", "D": "formal, ``I am a Democrat''", "EL": "formal, ``I am Mexican-American''",
            "IL": "casual, light Spanglish, San Antonio", "HL": "casual, Spanglish, ``I'm Mexican-American''",
            "EP": "formal, ``I am Polish-American''", "IP": "casual, babcia and pierogi, Buffalo", "HP": "casual, babcia, ``I'm Polish-American''"}
HEADLINE_SCRIPT = ["enclave_offered_latino_list", "enclave_offered_polish_list", "hispanic_share_mean", "white_nh_share_mean",
                   "black_nh_share_mean", "n_neighborhoods", "offrail_share", "commute_fail_share", "l2_tokens_count",
                   "l2_greeting_signoff", "political_descriptors_count", "identity_terms_count", "identity_in_opening",
                   "over_limit_share", "top_pick_over_limit", "adjacent_to_imd_offered", "enclave_epithet", "family_section_share",
                   "word_count", "rent_figures_count", "exclamation_density"]
ROUND3 = ["identity_fit_share", "absence_as_con", "coethnic_ranking_stated", "final_pick_identity_justified", "identity_outweighs_constraint",
          "coethnic_presence_fabricated", "enclave_as_errand", "family_script", "family_reassurance_is_safety", "acceptance_reassured",
          "spec_rewritten", "verdict_given", "overrun_minimized", "own_group_immersion", "area_identity_as_caution", "own_identity_cautioned",
          "over_limit_share", "top_pick_over_limit", "adjacent_to_imd_offered", "enclave_epithet", "family_section_share"]

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

def rates_longtable(summ, contr, axes_ids, label, caption, kinds, run, labels=None):
    """Like rates_table, as a longtable that may break across pages; rows in the given order with optional family headers (ids starting with '#')."""
    labels = labels or {}
    piv = summ.pivot(index="axis", columns="arm", values="mean").reindex([a for a in axes_ids if not a.startswith("#")])[ORDER]
    holm = contr.pivot(index="axis", columns="arm", values="p_holm").reindex(piv.index) if len(contr) else None
    head = "\\textbf{axis} & " + " & ".join(f"\\textbf{{{a}}}" for a in ORDER) + " \\\\"
    lines = [header(run, caption.split(".")[0]), "\\begin{center}\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{longtable}{@{}l" + "r" * len(ORDER) + "@{}}",
             f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
             "\\toprule " + head + " \\midrule \\endfirsthead",
             "\\toprule " + head + " \\midrule \\endhead",
             "\\midrule \\multicolumn{" + str(len(ORDER) + 1) + "}{r}{\\footnotesize $^{*}$ differs from the formal baseline, permutation test, Holm-adjusted within the arm, $p<0.05$} \\\\ \\endfoot",
             "\\bottomrule \\multicolumn{" + str(len(ORDER) + 1) + "}{r}{\\footnotesize $^{*}$ differs from the formal baseline, permutation test, Holm-adjusted within the arm, $p<0.05$} \\\\ \\endlastfoot"]
    for ax in axes_ids:
        if ax.startswith("#"):
            lines.append(f"\\multicolumn{{{len(ORDER) + 1}}}{{@{{}}l}}{{\\textbf{{{tex_escape(ax[1:])}}}}} \\\\"); continue
        if ax not in piv.index: continue
        cells = []
        for arm in ORDER:
            v = piv.loc[ax, arm]; s_ = fmt(v, kinds.get(ax, "rate"))
            if holm is not None and arm != "F" and ax in holm.index:
                p_ = holm.loc[ax, arm]
                if isinstance(p_, float) and not np.isnan(p_) and p_ < 0.05 and s_ != "--": s_ += "$^{*}$"
            cells.append(s_)
        lines.append(tex_escape(labels.get(ax, ax.replace("_", " "))) + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\end{longtable}", "\\end{center}", ""]
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
           "Judge protocol & eight batched calls per response, grouped by axis family, the rubric beside each question; evidence quote then yes/no, JSON schema output \\\\",
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
    keep = moving_axes(summ)
    judge_ids = [a["id"] for a in JUDGE_AXES if a["id"] in set(summ.axis) and a["id"] in keep]
    script_ids = [a["id"] for a in AXES if a["kind"] == "script" and a["id"] in set(summ.axis) and a["id"] in keep]
    pct = {"hispanic_share_mean": "pct", "white_nh_share_mean": "pct", "black_nh_share_mean": "pct", "median_rent_mean": "pct"}
    ints = {k: "int" for k in ("n_neighborhoods", "l2_tokens_count", "word_count", "sentence_count", "rent_figures_count", "political_descriptors_count",
                               "exclamation_density", "identity_terms_count", "cta_lines_count", "stations_count", "streets_landmarks_count",
                               "listing_sites_count", "urls_count", "structure_count", "caveat_density", "euphemism_density", "questions_count",
                               "mock_language_markers", "echo_user_l2", "casual_lexicon_rate", "emoji_count", "contraction_rate", "endearment_vocative",
                               "comprehension_checks", "warmth_markers", "fk_grade")}
    kinds = {**pct, **ints}
    (out / "steering_rates.tex").write_text(rates_longtable(summ, contr, ordered(judge_ids), "tab:steering-rates",
        f"Judge axes whose rate ranges by at least 0.2 across the arms: ensemble yes rate, three judges averaged, ${n_per_arm}$ responses per arm, grouped by family and ordered within a family by the largest gap from the baseline. Identity fit share is a share in $[0,1]$ averaged over the judges that wrote a count.", kinds, args.run))
    (out / "steering_service.tex").write_text(rates_longtable(summ, contr, ordered(script_ids), "tab:steering-service",
        "Measured axes that move across the arms (a range of at least 0.2 for shares, of at least 20 percent of the largest value for counts and census percentages): shares of responses or of offered areas in $[0,1]$, mean census shares of the offered community areas in percent, and counts, grouped by family and ordered within a family by the largest gap from the baseline.", kinds, args.run))
    # agreement over every judge axis
    ag = agree[~agree.axis.str.endswith("@r1")].set_index("axis")
    ag_ids = [k for k in ordered(judge_ids) if k.startswith("#") or k in ag.index]
    al = [header(args.run, "Judge agreement"), "\\begin{center}\\footnotesize", "\\begin{longtable}{@{}lrrrrr@{}}",
          f"\\caption{{Judge agreement on the judge axes of \\autoref{{tab:steering-rates}}: Fleiss' $\\kappa$ and pairwise agreement across the three judges, and each judge's yes rate pooled over arms. Mean $\\kappa$ over these axes: {ag.loc[[k for k in judge_ids if k in ag.index]].fleiss_kappa.mean():.2f}; mean pairwise agreement: {ag.loc[[k for k in judge_ids if k in ag.index]].pairwise_agreement.mean():.2f}.}}\\label{{tab:steering-agreement}}\\\\",
          "\\toprule \\textbf{axis} & $\\kappa$ & \\textbf{agree} & \\textbf{Haiku 5.5} & \\textbf{GPT-5-nano} & \\textbf{Qwen 3.7 Flash} \\\\ \\midrule \\endfirsthead",
          "\\toprule \\textbf{axis} & $\\kappa$ & \\textbf{agree} & \\textbf{Haiku 5.5} & \\textbf{GPT-5-nano} & \\textbf{Qwen 3.7 Flash} \\\\ \\midrule \\endhead",
          "\\bottomrule \\endfoot"]
    for ax in ag_ids:
        if ax.startswith("#"):
            al.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textbf{{{tex_escape(ax[1:])}}}}} \\\\"); continue
        row = ag.loc[ax]
        f2 = lambda v: "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.2f}"
        al.append(f"{tex_escape(ax.replace('_',' '))} & {f2(row.fleiss_kappa)} & {f2(row.pairwise_agreement)} & {f2(row.get('yes_haiku55', float('nan')))} & {f2(row.get('yes_nano', float('nan')))} & {f2(row.get('yes_qwen', float('nan')))} \\\\")
    al += ["\\end{longtable}", "\\end{center}", ""]
    (out / "steering_agreement.tex").write_text("\n".join(al))
    # catalogue
    cl = [header(args.run, "The candidate axes"), "\\begin{center}\\scriptsize",
          "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{0.21\\linewidth}l>{\\raggedright\\arraybackslash}p{0.66\\linewidth}@{}}",
          f"\\caption{{The {len(AXES)} behavior axes. Kind: J = yes/no answered by the judge ensemble, M = measured from the text and the pinned tables. The judge rubrics, lexicons and tables are in the pipeline repository.}}\\label{{tab:steering-axes}}\\\\",
          "\\toprule \\textbf{axis} & \\textbf{kind} & \\textbf{question} \\\\ \\midrule \\endfirsthead",
          "\\toprule \\textbf{axis} & \\textbf{kind} & \\textbf{question} \\\\ \\midrule \\endhead", "\\bottomrule \\endfoot"]
    for fam in ("steering", "identity", "constraints", "preferences", "service", "form"):
        cl.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\textbf{{{tex_escape(fam)}}}}} \\\\")
        for a in [x for x in AXES if x["family"] == fam]:
            cl.append(f"{tex_escape(a['id'].replace('_',' '))} & {'J' if a['kind']=='judge' else 'M'} & {tex_escape(a['q'])} \\\\")
    cl += ["\\end{longtable}", "\\end{center}", ""]
    (out / "steering_axes_catalogue.tex").write_text("\n".join(cl))
    print("tables ->", out)

if __name__ == "__main__":
    main()
