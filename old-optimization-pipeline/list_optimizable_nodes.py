#!/usr/bin/env python3
"""Extract nodes with long travel times and surface optimisation candidates."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd


TRAVEL_TIME_COLUMN = "travel_time_min"
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Go up from optimization-pipeline/ to root
DEFAULT_INPUT = PROJECT_ROOT / "data" / "analysis" / "nodes_with_scores.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "optimization" / "high_travel_time_nodes.csv"
DEFAULT_THRESHOLD = 15.0
DEFAULT_COLUMNS: Sequence[str] = (
    "walkability",
    "accessibility_score",
    "equity_score",
    TRAVEL_TIME_COLUMN,
)


def resolve_columns(frame: pd.DataFrame, requested: Sequence[str]) -> Sequence[str]:
    """Keep only columns that exist in the source frame."""
    return [column for column in requested if column in frame.columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter nodes_with_scores for long travel times.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to nodes_with_scores.parquet (defaults to <project>/data/analysis).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV file to write the filtered nodes list.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Travel time threshold in minutes (default: 15).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of rows written (after sorting).",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default=TRAVEL_TIME_COLUMN,
        help=f"Metric used to order optimisation candidates (default: {TRAVEL_TIME_COLUMN}).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"nodes_with_scores parquet not found: {args.input}")

    frame = pd.read_parquet(args.input)
    if TRAVEL_TIME_COLUMN not in frame.columns:
        raise ValueError(
            f"Column '{TRAVEL_TIME_COLUMN}' missing from dataset; cannot evaluate travel times."
        )

    travel_minutes = pd.to_numeric(frame[TRAVEL_TIME_COLUMN], errors="coerce")
    optimisable = frame.loc[travel_minutes > args.threshold].copy()
    optimisable[TRAVEL_TIME_COLUMN] = travel_minutes[travel_minutes > args.threshold]

    sort_column = args.sort_by or TRAVEL_TIME_COLUMN
    if sort_column in optimisable.columns:
        ascending = False if sort_column == TRAVEL_TIME_COLUMN else True
        optimisable = optimisable.sort_values(sort_column, ascending=ascending)

    selected_columns = resolve_columns(optimisable, DEFAULT_COLUMNS)
    if selected_columns:
        optimisable = optimisable[selected_columns]

    if args.limit is not None and args.limit > 0:
        optimisable = optimisable.head(args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    optimisable.to_csv(args.output, index=True)

    print(
        "Filtered %d nodes above %.2f minutes; wrote %d rows to %s" % (
            len(optimisable),
            args.threshold,
            len(optimisable),
            args.output,
        )
    )


if __name__ == "__main__":
    main()
