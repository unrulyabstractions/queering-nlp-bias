"""Import pre-generated text from CSV into the generation.json format.

Converts a CSV dataset of LLM generations into the generation.json format
expected by the scoring, estimation, and visualization pipeline stages.
No model is run — all scoring is text-based only.

Usage:
    uv run python scripts/import_csv_generations.py data.csv
    uv run python scripts/import_csv_generations.py data.csv --output out/myexp/generation.json
    uv run python scripts/import_csv_generations.py data.csv \\
        --text-col 0 --label-col 1 --model gpt-4o --prompt "Write a sentence."

CSV format (default):
    Column 0: generated text
    Column 1: arm/condition label (e.g. "man", "woman")
    Optional header row is skipped by default.

    Example:
        text,label
        "He fixed the car quickly.",man
        "She fixed the car quickly.",woman

Outputs:
    out/<csv_stem>/generation.json   (default, matches pipeline convention)
    out/<csv_stem>/generation_cfg.json  (stub config so viz loads metadata)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.csv_import_builder import (
    CsvImportConfig,
    CsvRow,
    build_generation_output_from_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Import pre-generated CSV text into generation.json format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.csv
  %(prog)s data.csv --output out/myexperiment/generation.json
  %(prog)s data.csv --text-col 0 --label-col 1 --no-header
  %(prog)s data.csv --model gpt-4o --prompt "Describe the patient."
""",
    )
    parser.add_argument("csv", help="Path to input CSV file")
    parser.add_argument(
        "--output", "-o",
        help="Output path for generation.json (default: out/<csv_stem>/generation.json)",
    )
    parser.add_argument(
        "--text-col",
        type=int,
        default=0,
        metavar="N",
        help="Zero-based column index for generated text (default: 0)",
    )
    parser.add_argument(
        "--label-col",
        type=int,
        default=1,
        metavar="N",
        help="Zero-based column index for arm/condition label (default: 1)",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="CSV has no header row (by default the first row is skipped)",
    )
    parser.add_argument(
        "--model",
        default="external",
        help="Model name to record in metadata (default: external)",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Prompt text to record in metadata (default: empty)",
    )
    return parser.parse_args()


def load_csv(
    path: Path,
    text_col: int,
    label_col: int,
    has_header: bool,
) -> list[CsvRow]:
    """Load rows from a CSV file."""
    rows: list[CsvRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)  # skip header
        for line_num, row in enumerate(reader, start=2 if has_header else 1):
            if len(row) <= max(text_col, label_col):
                print(
                    f"Warning: line {line_num} has only {len(row)} columns, skipping",
                    file=sys.stderr,
                )
                continue
            text = row[text_col].strip()
            label = row[label_col].strip()
            if not text or not label:
                print(
                    f"Warning: line {line_num} has empty text or label, skipping",
                    file=sys.stderr,
                )
                continue
            rows.append(CsvRow(text=text, label=label))
    return rows


def write_stub_generation_cfg(output_path: Path, model: str, prompt: str) -> None:
    """Write a generation_cfg.json stub so visualizations can load metadata."""
    cfg = {"model": model, "prompt": prompt, "trunk": "", "branches": []}
    cfg_path = output_path.parent / "generation_cfg.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def main() -> None:
    """Parse arguments and run import."""
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    output_path = (
        Path(args.output)
        if args.output
        else Path("out") / csv_path.stem / "generation.json"
    )

    rows = load_csv(
        csv_path,
        text_col=args.text_col,
        label_col=args.label_col,
        has_header=not args.no_header,
    )
    if not rows:
        print("Error: no valid rows found in CSV", file=sys.stderr)
        sys.exit(1)

    config = CsvImportConfig(model=args.model, prompt=args.prompt)
    output = build_generation_output_from_csv(rows, config)
    output.save(output_path)

    write_stub_generation_cfg(output_path, model=args.model, prompt=args.prompt)

    label_counts: dict[str, int] = {}
    for row in rows:
        label_counts[row.label] = label_counts.get(row.label, 0) + 1

    print(f"Imported {len(rows)} rows across {len(label_counts)} arms:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    print(f"Saved to {output_path}")
    print(f"\nNext steps:")
    print(f"  uv run python scripts/score_estimate_visualize.py {output_path} <scoring.json>")


if __name__ == "__main__":
    main()
