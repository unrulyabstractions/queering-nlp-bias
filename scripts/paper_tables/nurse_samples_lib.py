"""
Build the extended-samples appendix table from the scoring JSON.

For each arm, pick K=10 trajectories so that the mean of their per-cell
ensemble scores is close to the arm's core. We do simple stratified-by-
signature sampling: bucket trajectories by their 4-tuple ensemble pattern,
then allocate samples proportional to each bucket's mass, rounding via a
largest-remainder rule. Within each bucket we take the lowest traj_idx.

Emits a LaTeX file with one longtable per arm. The full trajectory is
included; gendered words and character names are bolded so the reader can
see at a glance what drove each judge's verdict. Score columns show three
per-judge verdicts (Opus / GPT-5 / Gemini) as colored cells; the cell
background reflects the ensemble mean.

Each emitted row carries an `% Source: ...` LaTeX comment with the
trajectory's source file, arm, and traj_idx for later verification.
"""
from __future__ import annotations
import json
import re
import os
from pathlib import Path

ROOT = Path("/Users/unrulyabstractions/work/OLD/queering-nlp-bias/out/nurse_sub/all")
SCORING_JSON = ROOT / "scoring.json"
# Per-arm default behaviors come from the same estimator file the main
# paper's Table 1 (tab:case-study-cores) reads, so the caption values and
# the table values agree to the last reported decimal. Deriving them from
# scoring.json instead drops the greedy trajectory (n = 200 vs n = 201) and
# shifts the third decimal.
ESTIMATION_JSON = ROOT / "estimation.json"
OUT = Path("/Users/unrulyabstractions/work/papers/homo/appendix/generated/extended_samples.tex")

ARMS = ["root", "trunk", "branch_1", "branch_2", "branch_3"]
ARM_TITLES = {
    "root": r"\texttt{root}: prompt only",
    "trunk": r"\texttt{trunk}: prompt + ``At the altar, the nurse and''",
    "branch_1": r"\texttt{branch\_1}: prompt + trunk + `` his partner''",
    "branch_2": r"\texttt{branch\_2}: prompt + trunk + `` her partner''",
    "branch_3": r"\texttt{branch\_3}: prompt + trunk + `` their partner''",
}
QUESTION_LABELS = [r"$B_{\text{male}}$", r"$B_{\text{female}}$",
                   r"$B_{\text{hetero}}$", r"$B_{\text{same-sex}}$"]
JUDGES = ["opus", "gpt5", "gemini"]
K = 3

ANSWER_RE = re.compile(r"ANSWER:\s*([01])")

# Curated lists of words to bold so the reader can see at a glance what the
# judges keyed off of. Keep the lists conservative; over-bolding reads as
# noise.
NAMES_TO_BOLD = {
    "Sarah", "Marcus", "David", "Daniel", "James", "Michael", "John",
    "Mark", "Matthew", "Luke", "Andrew", "Tom", "Tomas", "Thomas",
    "Maya", "Emma", "Lily", "Anna", "Anne", "Lisa", "Karen", "Maria",
    "Elena", "Olivia", "Sophie", "Sophia", "Catherine", "Rebecca",
    "Hannah", "Amelia", "Grace", "Ella", "Eva", "Eve", "Eli", "Ben",
    "Alex", "Jordan", "Sam", "Chris", "Pat", "Jamie", "Robert", "Carlos",
    "Diego", "Javier", "Lucas", "Ethan", "Liam", "Noah", "Oliver",
    "William", "Theo", "Theodore", "Henry", "Edward", "Patrick",
    "Jennifer", "Jessica", "Rachel", "Megan", "Amanda", "Christina",
    "Nicole", "Samantha", "Isabella", "Ava", "Mia", "Charlotte",
    "Amelie", "Claire", "Diana", "Stella", "Nina", "Nora", "Ruth",
    "Jose", "Antonio", "Miguel", "Pedro", "Ricardo", "Rafael", "Pablo",
    "Yuki", "Kenji", "Hiroshi", "Mei", "Lin", "Jun",
}
PRONOUNS_FEM = {"she", "her", "hers", "herself", "She", "Her", "Hers", "Herself"}
PRONOUNS_MASC = {"he", "his", "him", "himself", "He", "His", "Him", "Himself"}
ROLE_WORDS = {
    "nurse", "Nurse", "nurses", "Nurses",
    "wife", "Wife", "husband", "Husband", "wives", "husbands",
    "girlfriend", "boyfriend", "Girlfriend", "Boyfriend",
    "bride", "groom", "Bride", "Groom",
}

WORDS_TO_BOLD = NAMES_TO_BOLD | PRONOUNS_FEM | PRONOUNS_MASC | ROLE_WORDS

# Build a single regex; sort by length so longer words match first.
BOLD_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(WORDS_TO_BOLD, key=len, reverse=True)) + r")\b"
)


def parse_verdict(text: str | None):
    if not text:
        return None
    m = ANSWER_RE.findall(text)
    if not m:
        return None
    return int(m[-1])


def latex_escape(s: str) -> str:
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}"}
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def cleanup(text: str) -> str:
    """Strip leading markdown headers and collapse whitespace; keep the
    full text (no truncation)."""
    t = text.lstrip()
    if t.startswith("#"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:].lstrip()
    t = t.replace("\n", " ").strip()
    # Collapse runs of whitespace.
    t = re.sub(r"\s+", " ", t)
    return t


def bold_keywords(escaped: str) -> str:
    """Wrap occurrences of curated words in \\textbf{...}.
    Run after latex_escape since the curated words contain only ASCII
    letters (no LaTeX-special chars), so escaping does not move them."""
    return BOLD_RE.sub(r"\\textbf{\1}", escaped)


def stratified_pick(items, k):
    """Pick k items so the mean of their score vectors is close to the
    arm's mean. Bucket by the 4-tuple signature; allocate counts by largest-
    remainder over bucket masses; pick lowest traj_idx per bucket."""
    buckets: dict[tuple, list] = {}
    for it in items:
        sig = tuple(it["scores"])
        buckets.setdefault(sig, []).append(it)
    n = len(items)
    raw = [(sig, len(b) * k / n) for sig, b in buckets.items()]
    floor = [(sig, int(r), r - int(r)) for sig, r in raw]
    base = sum(f for _, f, _ in floor)
    leftover = k - base
    floor.sort(key=lambda t: -t[2])
    counts = {sig: f for sig, f, _ in floor}
    for sig, _, _ in floor[: leftover]:
        counts[sig] = counts.get(sig, 0) + 1
    picks = []
    for sig, cnt in counts.items():
        if cnt == 0:
            continue
        bucket = sorted(buckets[sig], key=lambda it: it["traj_idx"])
        picks.extend(bucket[:cnt])
    picks.sort(key=lambda it: (it["arm"], it["traj_idx"]))
    return picks


def cell_verdict(verdicts):
    """Render a triple (opus, gpt5, gemini) as 3 colored mini-boxes."""
    out = []
    for v in verdicts:
        if v == 1:
            out.append(r"\verdictyes")
        elif v == 0:
            out.append(r"\verdictno")
        else:
            out.append(r"\verdictna")
    return r"\,".join(out)


def cell_color(score):
    # score in [0,1] -> background tint. Use a single neutral hue (gray->amber)
    # so the table reads as intensity rather than category.
    pct = int(round(score * 60))  # 0..60
    return rf"\cellcolor{{cellShade!{pct}}}"


def build_arm_table(arm, items, core, prefill):
    rows = []
    for it in items:
        # Prepend the arm's prefill so the trajectory is shown in full
        # from the very beginning. Render the prefill in gray italics so
        # the reader can see where the model started sampling.
        cont = cleanup(it["text"])
        if prefill:
            pre_esc = latex_escape(prefill)
            pre_bold = bold_keywords(pre_esc)
            prefill_tex = rf"\textcolor{{black!55}}{{{pre_bold}}}"
            cont_esc = latex_escape(cont)
            cont_bold = bold_keywords(cont_esc)
            # cleanup() lstripped any leading whitespace from the
            # continuation, so insert a single space between prefill
            # and continuation.
            bolded = prefill_tex + " " + cont_bold
        else:
            escaped = latex_escape(cont)
            bolded = bold_keywords(escaped)
        per_judge_str = ", ".join(
            f"{j}={','.join(str(v) if v is not None else 'NA' for v in it['per_judge'][j])}"
            for j in JUDGES
        )
        comment = (
            f"% Source: scoring.json results[arm={arm}, traj_idx={it['traj_idx']}]\n"
            f"%   text from generation.json[\"{arm}\"][{it['traj_idx']}]\n"
            f"%   per-judge (male,female,hetero,same-sex): {per_judge_str}"
        )
        cells = [rf"\scriptsize\texttt{{{it['traj_idx']:03d}}}",
                 rf"\scriptsize\itshape {bolded}"]
        for q in range(4):
            triple = [it["per_judge"][j][q] for j in JUDGES]
            score = sum(v for v in triple if v is not None) / max(
                1, sum(1 for v in triple if v is not None))
            cell = f"{cell_color(score)}{cell_verdict(triple)}"
            cells.append(cell)
        rows.append(comment + "\n" + " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    core_str = " / ".join(f"{c:.3f}" for c in core)
    title = ARM_TITLES[arm]
    cap = (rf"Extended generation samples for {title}. "
           rf"Default behavior (\textit{{male / female / hetero / same-sex}}): {core_str}. "
           rf"Each score column shows three judge verdicts in order "
           rf"(\textsc{{Opus}}, \textsc{{GPT-5}}, \textsc{{Gemini}}). "
           rf"A green box (\verdictyes) is a yes verdict, a red box "
           rf"(\verdictno) is a no verdict, and an empty outline "
           rf"(\verdictna) marks an unparseable response. "
           rf"Cell shading reflects the ensemble mean. "
           rf"In the trajectory column, character names, gendered pronouns, and role words are bolded.")
    return rf"""
\renewcommand{{\arraystretch}}{{1.10}}
\setlength{{\tabcolsep}}{{4pt}}
\begin{{longtable}}{{@{{}}p{{0.9cm}}p{{10.8cm}}cccc@{{}}}}
\caption{{{cap}}}
\label{{tab:samples-{arm}}}\\
\toprule
\textbf{{Idx}} & \textbf{{Trajectory}} & {' & '.join(QUESTION_LABELS)} \\
\midrule
\endfirsthead
\toprule
\textbf{{Idx}} & \textbf{{Trajectory}} & {' & '.join(QUESTION_LABELS)} \\
\midrule
\endhead
\bottomrule
\endfoot
{body}
\bottomrule
\end{{longtable}}
"""


def main():
    # The original judges (opus, gpt5, gemini) store their reply texts in the fixed order q3, q2, q0, q1, so parsing
    # replies by slot (below) mis-slots their verdicts. Use scripts/paper_tables/nurse_tables.py, which reads method_scores.
    raise SystemExit("nurse_samples_lib.main() is retired: run scripts/paper_tables/nurse_tables.py instead")
    with open(SCORING_JSON) as f:
        data = json.load(f)

    # Build per-trajectory records with parsed per-judge verdicts.
    rows_by_arm: dict[str, list[dict]] = {a: [] for a in ARMS}
    for r in data["results"]:
        arm = r["arm"]
        if arm not in rows_by_arm:
            continue
        raw = r["method_raw"]["categorical"]
        per_judge = {j: [parse_verdict(t) for t in raw.get(j, [None] * 4)]
                     for j in JUDGES}
        rows_by_arm[arm].append({
            "arm": arm,
            "traj_idx": r["traj_idx"],
            "text": r["text"],
            "scores": [round(s, 3) for s in r["method_scores"]["categorical"]],
            "per_judge": per_judge,
        })

    # Per-arm default behaviors: read the uniform-mean core straight out of
    # estimation.json, which is the same source the main paper's Table 1
    # (tab:case-study-cores) reports, so caption and table agree exactly.
    with open(ESTIMATION_JSON) as f:
        est = json.load(f)
    cores = {}
    for arm_rec in est["arms"]:
        name = arm_rec["name"]
        if name in rows_by_arm:
            cores[name] = list(arm_rec["estimates"]["uniform"]["core"])
    missing = [a for a in ARMS if a not in cores]
    if missing:
        raise SystemExit(f"no uniform core in {ESTIMATION_JSON} for arms: {missing}")

    parts = [
        r"% Auto-generated by appendix/generated/build_samples.py.",
        r"% Do not edit by hand; regenerate via `python build_samples.py`.",
        r"%",
        rf"% Source: {SCORING_JSON}",
        rf"% Trajectory text from /Users/unrulyabstractions/work/OLD/queering-nlp-bias/out/nurse_sub/generation.json",
        r"% Per-row comments record arm + traj_idx + per-judge verdicts so a",
        r"% reader can audit each entry against the scoring JSON.",
        r"",
    ]
    # Pull per-arm prefill text from the generation config so we can
    # show the full trajectory (prefill + sampled continuation).
    gen_cfg = json.load(open(
        "/Users/unrulyabstractions/work/OLD/queering-nlp-bias/out/nurse_sub/generation.json"
    ))["config"]
    prefills = {a["name"]: a.get("prefill", "") for a in gen_cfg["arms"]}

    for arm in ARMS:
        items = rows_by_arm[arm]
        picks = stratified_pick(items, K)
        parts.append(build_arm_table(arm, picks, cores[arm], prefills.get(arm, "")))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts))
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes, {sum(len(rows_by_arm[a]) for a in ARMS)} trajectories scanned)")


if __name__ == "__main__":
    main()
