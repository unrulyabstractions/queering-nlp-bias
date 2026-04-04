"""Merge specified and associated CSVs into a two-column (text, label) CSV.

Reads two CSVs produced by the More-of-the-Same dataset pipeline and outputs
a combined CSV (no header) with columns: text, gender_label.

  Specified CSV  — uses the "text" and "gender" columns; labels get
                   " (specified)" appended.
  Associated CSV — uses the "text" and "inferred_gender" columns; labels get
                   " (associated)" appended.

Sampling (applied independently to each CSV):
  For each (occupation, gender) group, up to --samples-per-gender rows are kept.
  For each occupation group, the total is further capped at --samples-per-occupation.
  When --random-sample is set (default), rows are drawn randomly; otherwise the
  first N rows in file order are taken.

Usage:
    uv run python scripts/import_more_of_the_same.py specified.csv associated.csv
    uv run python scripts/import_more_of_the_same.py specified.csv associated.csv \\
        --output out/mots/merged.csv
    uv run python scripts/import_more_of_the_same.py specified.csv associated.csv \\
        --samples-per-occupation 5 --samples-per-gender 2 --no-random-sample    uv run python scripts/import_more_of_the_same.py specified.csv associated.csv --all-pairs
    uv run python scripts/import_more_of_the_same.py specified.csv associated.csv \
        --all-pairs --exclude-non-binary"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

_TRIALS_CSV_DIR = Path(__file__).parent.parent / "trials" / "csv_import_data"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge specified + associated CSVs into a two-column (text, label) CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s specified.csv associated.csv
  %(prog)s specified.csv associated.csv --output out/mots/merged.csv
  %(prog)s specified.csv associated.csv --samples-per-occupation 5 --samples-per-gender 2
  %(prog)s specified.csv associated.csv --no-random-sample
""",
    )
    parser.add_argument("specified_csv", help="Path to the 'specified' CSV (has 'gender' column)")
    parser.add_argument("associated_csv", help="Path to the 'associated' CSV (has 'inferred_gender' column)")
    parser.add_argument(
        "--output", "-o",
        help="Output CSV path (default: trials/csv_import_data/more-of-the-same-<pg>-pg-<po>-po[-random].csv)",
    )
    parser.add_argument(
        "--samples-per-occupation",
        type=int,
        default=3,
        metavar="N",
        help="Max rows per occupation after gender sampling (default: 3)",
    )
    parser.add_argument(
        "--samples-per-gender",
        type=int,
        default=3,
        metavar="N",
        help="Max rows per (occupation, gender) group (default: 3)",
    )
    parser.add_argument(
        "--random-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample rows randomly within each group (default: True; use --no-random-sample for "
             "deterministic first-N selection)",
    )
    parser.add_argument(
        "--exclude-non-binary",
        action="store_true",
        default=False,
        help="Exclude rows where the gender marker is 'N' (non-binary)",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        default=False,
        help="Keep all occupation/gender pairs without any sampling or count equalization",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_specified_csv(path: Path) -> list[tuple[str, str, str]]:
    """Return (text, gender, occupation) triples from the specified CSV."""
    rows: list[tuple[str, str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            gender = (row.get("gender") or "").strip()
            occupation = (row.get("occupation") or "").strip()
            if not text or not gender:
                continue
            rows.append((text, gender, occupation))
    return rows


def load_associated_csv(path: Path) -> list[tuple[str, str, str]]:
    """Return (text, inferred_gender, occupation) triples from the associated CSV."""
    rows: list[tuple[str, str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            gender = (row.get("inferred_gender") or "").strip()
            occupation = (row.get("occupation") or "").strip()
            if not text or not gender:
                continue
            rows.append((text, gender, occupation))
    return rows


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample(pool: list, n: int, use_random: bool) -> list:
    """Return up to n items from pool, randomly or from the front."""
    if len(pool) <= n:
        return pool
    if use_random:
        return random.sample(pool, n)
    return pool[:n]


def sample_rows(
    rows: list[tuple[str, str, str]],
    samples_per_gender: int,
    samples_per_occupation: int,
    use_random: bool,
) -> list[tuple[str, str, str]]:
    """Apply two-level sampling: per (occupation, gender) then per occupation."""
    # Group by (occupation, gender)
    occ_gender_groups: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for row in rows:
        text, gender, occupation = row
        key = (occupation, gender)
        occ_gender_groups.setdefault(key, []).append(row)

    # First pass: cap per (occupation, gender)
    after_gender_cap: dict[str, list[tuple[str, str, str]]] = {}
    for (occupation, gender), group in occ_gender_groups.items():
        kept = _sample(group, samples_per_gender, use_random)
        after_gender_cap.setdefault(occupation, []).extend(kept)

    # Second pass: cap per occupation
    result: list[tuple[str, str, str]] = []
    for occupation, group in after_gender_cap.items():
        result.extend(_sample(group, samples_per_occupation, use_random))

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_output(
    specified_rows: list[tuple[str, str, str]],
    associated_rows: list[tuple[str, str, str]],
    output_path: Path,
) -> None:
    """Write the merged two-column (text, label) CSV to a file."""
    out_rows: list[tuple[str, str]] = []

    for text, gender, _occ in specified_rows:
        out_rows.append((text, f"{gender} (specified)"))

    for text, gender, _occ in associated_rows:
        out_rows.append((text, f"{gender} (associated)"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    specified_path = Path(args.specified_csv)
    associated_path = Path(args.associated_csv)

    for p in (specified_path, associated_path):
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    specified_rows = load_specified_csv(specified_path)
    associated_rows = load_associated_csv(associated_path)

    if args.exclude_non_binary:
        specified_rows = [(t, g, o) for t, g, o in specified_rows if g.upper() != "N"]
        associated_rows = [(t, g, o) for t, g, o in associated_rows if g.upper() != "N"]

    if args.all_pairs:
        specified_sampled = specified_rows
        associated_sampled = associated_rows
    else:
        specified_sampled = sample_rows(
            specified_rows,
            args.samples_per_gender,
            args.samples_per_occupation,
            args.random_sample,
        )
        associated_sampled = sample_rows(
            associated_rows,
            args.samples_per_gender,
            args.samples_per_occupation,
            args.random_sample,
        )

        # Equalise counts so specified and associated contribute equally.
        n = min(len(specified_sampled), len(associated_sampled))
        specified_sampled = _sample(specified_sampled, n, args.random_sample)
        associated_sampled = _sample(associated_sampled, n, args.random_sample)

    if args.output:
        output_path = Path(args.output)
    elif args.all_pairs:
        nb_suffix = "-no-nb" if args.exclude_non_binary else ""
        output_path = _TRIALS_CSV_DIR / f"more-of-the-same-all{nb_suffix}.csv"
    else:
        random_suffix = "-random" if args.random_sample else ""
        nb_suffix = "-no-nb" if args.exclude_non_binary else ""
        stem = (
            f"more-of-the-same"
            f"-{args.samples_per_gender}-pg"
            f"-{args.samples_per_occupation}-po"
            f"{random_suffix}"
            f"{nb_suffix}"
        )
        output_path = _TRIALS_CSV_DIR / f"{stem}.csv"
    write_output(specified_sampled, associated_sampled, output_path)

    total = len(specified_sampled) + len(associated_sampled)
    print(
        f"Wrote {total} rows "
        f"({len(specified_sampled)} specified, {len(associated_sampled)} associated)"
        f" → {output_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
