"""Helpers for the local-bias sample tables: LaTeX escaping, keyword bolding, stratified picks, verdict cells, rounding."""
from __future__ import annotations
import re

def hu(x, nd=3):
    """Round half up after clearing float noise, so one value prints the same in every table."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(round(float(x), 5))).quantize(Decimal(1).scaleb(-nd), ROUND_HALF_UP))



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
    """One colored mini-box per judge."""
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
