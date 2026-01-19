import argparse
import sys
import pandas as pd
from pathlib import Path
from typing import Sequence

# Add project root to path for CityDataManager
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from city_paths import CityDataManager


TRAVEL_TIME_COLUMN = "travel_time_min"
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
    
    # Core execution flags
    parser.add_argument("--city", default="bangalore", help="City to process")
    parser.add_argument("--mode", default="ga_only", choices=["ga_only", "ga_milp", "ga_milp_pnmlr"], help="Optimization mode")
    parser.add_argument("--force", action="store_true", help="Force recomputation")
    
    args = parser.parse_args()
    
    # Initialize CDM and load config
    cdm = CityDataManager(args.city, project_root=project_root, mode=args.mode)
    cfg = cdm.load_config()
    
    # Resolve paths
    input_path = cdm.baseline_nodes
    output_path = cdm.high_travel_nodes(args.mode)
    
    # Get settings from config
    opt_cfg = cfg.get('optimization', {})
    threshold = opt_cfg.get('candidate_threshold', 15.0)
    limit = opt_cfg.get('candidate_limit')
    
    print(f"Extracting candidates for {args.city} ({args.mode})")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Threshold: {threshold} min")
    if limit: print(f"  Limit: {limit} rows")

    if not input_path.exists():
        raise FileNotFoundError(f"nodes_with_scores parquet not found: {input_path}")

    frame = pd.read_parquet(input_path)
    if TRAVEL_TIME_COLUMN not in frame.columns:
        raise ValueError(f"Column '{TRAVEL_TIME_COLUMN}' missing from dataset.")

    travel_minutes = pd.to_numeric(frame[TRAVEL_TIME_COLUMN], errors="coerce")
    optimisable = frame.loc[travel_minutes > threshold].copy()
    optimisable[TRAVEL_TIME_COLUMN] = travel_minutes[travel_minutes > threshold]

    # Sort and filter
    optimisable = optimisable.sort_values(TRAVEL_TIME_COLUMN, ascending=False)
    selected_columns = resolve_columns(optimisable, DEFAULT_COLUMNS)
    if selected_columns:
        optimisable = optimisable[selected_columns]

    if limit is not None and limit > 0:
        optimisable = optimisable.head(limit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimisable.to_csv(output_path, index=True)

    print(f"Successfully wrote {len(optimisable)} candidates to {output_path}")


if __name__ == "__main__":
    main()
