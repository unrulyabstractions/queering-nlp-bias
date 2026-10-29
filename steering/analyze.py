"""Merge judgments and script axes; per-arm rates with CIs; contrasts against the formal baseline; judge agreement.

Usage: uv run python steering/analyze.py --run pilot [--perms 5000]
Writes steering/out/<run>/analysis/{table.csv, summary.csv, contrasts.csv, agreement.csv, summary.json}
and prints a pilot report: the axes with the largest gaps between any identity arm and F.
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from axes import AXES, JUDGE_AXES, SCRIPT_AXES
from prompts import ORDER, ARMS

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261029)
SCRIPT_BINARY = {"enclave_offered", "enclave_first", "cross_enclave_offered", "westside_black_area_named", "travel_time_stated",
                 "l2_greeting_signoff", "full_switch", "enclave_offered_latino_list", "enclave_offered_polish_list", "table_present", "congrats_opener",
                 "identity_in_opening"}
SCRIPT_NUMERIC = ["enclave_share", "hispanic_share_mean", "white_nh_share_mean", "black_nh_share_mean", "polish_heritage_share",
                  "offrail_share", "commute_fail_share", "rent_over_budget_share", "median_rent_mean", "n_neighborhoods",
                  "euphemism_density", "questions_count", "word_count", "sentence_count", "fk_grade", "rent_figures_count",
                  "cta_lines_count", "stations_count", "streets_landmarks_count", "listing_sites_count", "urls_count",
                  "structure_count", "caveat_density", "l2_tokens_count", "mock_language_markers", "echo_user_l2",
                  "casual_lexicon_rate", "emoji_count", "exclamation_density", "contraction_rate", "endearment_vocative",
                  "comprehension_checks", "warmth_markers", "style_matching", "political_descriptors_count", "identity_terms_count"]

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def boot_ci(x, B=2000):
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(x) == 0: return (float("nan"), float("nan"), 0)
    if len(x) == 1: return (x[0], x[0], 1)
    idx = RNG.integers(0, len(x), size=(B, len(x))); means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), int(len(x)))

def perm_p(a, b, n_perm):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return float("nan")
    obs = a.mean() - b.mean(); pool = np.concatenate([a, b]); na = len(a); cnt = 0
    for _ in range(n_perm):
        RNG.shuffle(pool); d = pool[:na].mean() - pool[na:].mean()
        if abs(d) >= abs(obs) - 1e-12: cnt += 1
    return (cnt + 1) / (n_perm + 1)

def fleiss_kappa(mat):
    """mat: n_items x n_raters of 0/1."""
    mat = np.asarray(mat); n, m = mat.shape
    if n == 0 or m < 2: return float("nan")
    k1 = mat.sum(axis=1); k0 = m - k1
    P_i = (k1 * (k1 - 1) + k0 * (k0 - 1)) / (m * (m - 1)); P_bar = P_i.mean()
    p1 = k1.sum() / (n * m); P_e = p1 ** 2 + (1 - p1) ** 2
    return float("nan") if P_e == 1 else (P_bar - P_e) / (1 - P_e)

def load(run_dir: Path):
    resp = {}
    for l in (run_dir / "responses.jsonl").open():
        if l.strip():
            d = json.loads(l); resp[d["id"]] = d
    judg = defaultdict(dict)  # id -> axis -> {judge: 0/1}; a refine batch overrides the first-round answer,
    judges = []               # which is kept under "<axis>@r1" for the before/after agreement table
    refine_axes = {a["id"] for a in JUDGE_AXES if str(a.get("batch", "")).startswith("refine")}
    for p in sorted((run_dir / "judgments").glob("*.jsonl")):
        j = p.stem; judges.append(j)
        recs = [json.loads(l) for l in p.open() if l.strip()]
        recs.sort(key=lambda d: str(d.get("batch", "")).startswith("refine"))  # first-round batches first
        for d in recs:
            refine = str(d.get("batch", "")).startswith("refine")
            for ax, v in d["answers"].items():
                vote = 1 if v["answer"] == "yes" else 0
                if ax in refine_axes and not refine:
                    judg[d["id"]].setdefault(ax + "@r1", {})[j] = vote
                    judg[d["id"]].setdefault(ax, {})[j] = vote  # overwritten below if a refine record exists
                else:
                    judg[d["id"]].setdefault(ax, {})[j] = vote
    script = {}
    sp = run_dir / "script_axes.jsonl"
    if sp.exists():
        for l in sp.open():
            if l.strip():
                d = json.loads(l); script[d["id"]] = d
    return resp, judg, judges, script

def build_table(resp, judg, judges, script):
    rows = []
    for rid, r in resp.items():
        row = {"id": rid, "arm": r["arm"], "group": r["group"], "signal": r["signal"]}
        for a in JUDGE_AXES:
            for aid in ([a["id"], a["id"] + "@r1"] if "rubric_round1" in a else [a["id"]]):
                votes = judg.get(rid, {}).get(aid, {})
                for j in judges:
                    row[f"{aid}__{j}"] = votes.get(j, np.nan)
                row[aid] = np.mean([votes[j] for j in judges if j in votes]) if votes else np.nan
        s = script.get(rid, {})
        for k in list(SCRIPT_BINARY) + SCRIPT_NUMERIC + ["switch_type", "first_area"]:
            v = s.get(k)
            row[k] = (np.nan if v is None else v)
        row["offered_areas"] = "|".join(s.get("offered_areas", []))
        rows.append(row)
    return pd.DataFrame(rows)

def arm_level(df):
    """set_divergence_from_f and within_arm_entropy."""
    out = {}
    f_sets = [set(x.split("|")) - {""} for x in df[df.arm == "F"].offered_areas]
    cnt = Counter(a for s in f_sets for a in s); modal = {a for a, c in cnt.items() if c >= 0.5 * max(1, len(f_sets))}
    def jacc(s):
        if not s and not modal: return 0.0
        return 1 - len(s & modal) / max(1, len(s | modal))
    df["set_divergence_from_f"] = [jacc(set(x.split("|")) - {""}) for x in df.offered_areas]
    for arm, g in df.groupby("arm"):
        firsts = Counter(x for x in g.first_area if isinstance(x, str)); n = sum(firsts.values())
        ent = -sum(c / n * math.log(c / n) for c in firsts.values()) if n else float("nan")
        alls = Counter(a for x in g.offered_areas for a in x.split("|") if a); m = sum(alls.values())
        ent_all = -sum(c / m * math.log(c / m) for c in alls.values()) if m else float("nan")
        out[arm] = {"entropy_first": ent, "entropy_all": ent_all, "modal_first": firsts.most_common(1)[0] if firsts else None,
                    "modal_first_share": (firsts.most_common(1)[0][1] / n) if n else float("nan"), "f_modal_set": sorted(modal)}
    return out

def summarize(df, judges, n_perm):
    axes_binary = [a["id"] for a in JUDGE_AXES] + sorted(SCRIPT_BINARY)
    axes_numeric = SCRIPT_NUMERIC + ["set_divergence_from_f"]
    srows, crows = [], []
    for ax in axes_binary + axes_numeric:
        if ax not in df: continue
        for arm in ORDER:
            x = df.loc[df.arm == arm, ax].astype(float)
            n = int(x.notna().sum()); mean = float(x.mean()) if n else float("nan")
            if ax in axes_binary and ax in {a["id"] for a in JUDGE_AXES}:
                lo, hi, _ = boot_ci(x.dropna().tolist())
            elif ax in axes_binary:
                lo, hi = wilson(int(round(x.sum())), n)
            else:
                lo, hi, _ = boot_ci(x.dropna().tolist())
            srows.append({"axis": ax, "arm": arm, "n": n, "mean": mean, "lo": lo, "hi": hi})
            if arm != "F":
                p = perm_p(x.dropna().values, df.loc[df.arm == "F", ax].astype(float).dropna().values, n_perm)
                crows.append({"axis": ax, "arm": arm, "diff_vs_F": mean - float(df.loc[df.arm == "F", ax].astype(float).mean()), "p": p})
    summ = pd.DataFrame(srows); contr = pd.DataFrame(crows)
    # Holm across all contrasts within each axis family? Report raw p and Holm over all judge+binary axes per arm.
    if len(contr):
        contr["p_holm"] = np.nan
        for arm, g in contr.groupby("arm"):
            ps = g["p"].values; order = np.argsort(ps); m = np.sum(~np.isnan(ps)); adj = np.full(len(ps), np.nan)
            running = 0.0
            for rank, i in enumerate(order):
                if np.isnan(ps[i]): continue
                val = min(1.0, (m - rank) * ps[i]); running = max(running, val); adj[i] = running
            contr.loc[g.index, "p_holm"] = adj
    # agreement
    arows = []
    for a in JUDGE_AXES:
        for aid in ([a["id"], a["id"] + "@r1"] if "rubric_round1" in a else [a["id"]]):
            cols = [f"{aid}__{j}" for j in judges if f"{aid}__{j}" in df]
            sub = df[cols].dropna()
            if len(sub) == 0 or len(cols) < 2: continue
            mat = sub.values.astype(int)
            pair = np.mean([np.mean(mat[:, i] == mat[:, k]) for i in range(mat.shape[1]) for k in range(i + 1, mat.shape[1])])
            arows.append({"axis": aid, "n": len(sub), "pairwise_agreement": pair, "fleiss_kappa": fleiss_kappa(mat),
                          "yes_rate_overall": float(mat.mean()), **{f"yes_{j}": float(sub[c].mean()) for j, c in zip(judges, cols)}})
    agree = pd.DataFrame(arows)
    return summ, contr, agree

def pilot_report(summ, contr, agree, arm_stats):
    piv = summ.pivot(index="axis", columns="arm", values="mean")[[c for c in ORDER if c in summ.arm.unique()]]
    ident = [c for c in piv.columns if c not in ("F", "D")]
    gap = (piv[ident].max(axis=1) - piv["F"]).abs() if ident else piv["F"] * 0
    gap2 = (piv[ident].min(axis=1) - piv["F"]).abs() if ident else gap
    piv["max_gap_vs_F"] = np.maximum(gap, gap2)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200); pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    print("\n=== per-arm means, sorted by the largest gap between an identity arm and F ===")
    print(piv.sort_values("max_gap_vs_F", ascending=False).to_string())
    print("\n=== arm-level ===")
    for arm, s in arm_stats.items():
        print(f"  {arm}: entropy_first={s['entropy_first']:.2f} entropy_all={s['entropy_all']:.2f} modal_first={s['modal_first']} share={s['modal_first_share']:.2f}")
    print("  F modal set:", arm_stats["F"]["f_modal_set"])
    if len(agree):
        print("\n=== judge agreement (lowest kappa first) ===")
        print(agree.sort_values("fleiss_kappa").head(20).to_string(index=False))
        print(f"  mean pairwise agreement {agree.pairwise_agreement.mean():.3f}; mean kappa {agree.fleiss_kappa.mean():.3f}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--perms", type=int, default=3000)
    args = ap.parse_args()
    run_dir = HERE / "out" / args.run; out = run_dir / "analysis"; out.mkdir(exist_ok=True)
    resp, judg, judges, script = load(run_dir)
    df = build_table(resp, judg, judges, script)
    arm_stats = arm_level(df)
    summ, contr, agree = summarize(df, judges, args.perms)
    df.to_csv(out / "table.csv", index=False); summ.to_csv(out / "summary.csv", index=False)
    contr.to_csv(out / "contrasts.csv", index=False); agree.to_csv(out / "agreement.csv", index=False)
    json.dump({"arm_stats": arm_stats, "judges": judges, "n_responses": len(df)}, (out / "summary.json").open("w"), indent=1, default=str)
    print(f"responses={len(df)} judges={judges} -> {out}")
    pilot_report(summ, contr, agree, arm_stats)

if __name__ == "__main__":
    main()
