#!/usr/bin/env python3
"""
Cache management utility for PathLens pipeline.
Helps clear specific cached data or view cache status.
"""
import argparse
from pathlib import Path
import shutil


def get_file_size(path: Path) -> str:
    """Get human-readable file size."""
    if not path.exists():
        return "N/A"
    
    if path.is_file():
        size = path.stat().st_size
    else:
        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def show_cache_status(project_dir: Path):
    """Display current cache status."""
    print("\n" + "=" * 60)
    print("📦 PathLens Cache Status")
    print("=" * 60)
    
    data_dir = project_dir / "data"
    scripts_dir = project_dir / "scripts"
    
    sections = [
        ("Raw OSM Data", [
            ("Street network", data_dir / "raw" / "graph.graphml"),
            ("Buildings", data_dir / "raw" / "buildings.geojson"),
            ("Land use", data_dir / "raw" / "landuse.geojson"),
            ("Transit", data_dir / "raw" / "transit.geojson"),
            ("POIs (converted)", data_dir / "raw" / "pois.geojson"),
            ("Nodes", data_dir / "raw" / "nodes.parquet"),
            ("Edges", data_dir / "raw" / "edges.parquet"),
        ]),
        ("Processed Data", [
            ("Simplified graph", data_dir / "processed" / "graph.graphml"),
            ("Processed nodes", data_dir / "processed" / "nodes.parquet"),
            ("Processed edges", data_dir / "processed" / "edges.parquet"),
            ("POI mapping", data_dir / "processed" / "poi_node_mapping.parquet"),
        ]),
        ("Analysis Results", [
            ("Node scores", data_dir / "analysis" / "nodes_with_scores.parquet"),
            ("Node scores CSV", data_dir / "analysis" / "nodes_with_scores.csv"),
            ("H3 aggregates", data_dir / "analysis" / "h3_agg.parquet"),
            ("H3 aggregates CSV", data_dir / "analysis" / "h3_agg.csv"),
            ("Metrics summary", data_dir / "analysis" / "metrics_summary.json"),
        ]),
        ("Visualizations", [
            ("Interactive map", scripts_dir / "interactive_map.html"),
        ]),
        ("Amenity Cache", [
            ("JSON files", scripts_dir / "cache" / "bengaluru_amenities"),
            ("Area list", scripts_dir / "cache" / "bengaluru_areas_list.json"),
        ]),
        ("OSMnx Cache", [
            ("API cache", scripts_dir / "cache"),
        ]),
    ]
    
    for section_name, items in sections:
        print(f"\n{section_name}:")
        for name, path in items:
            if path.exists():
                size = get_file_size(path)
                if path.is_dir():
                    file_count = len(list(path.rglob('*.json')))
                    print(f"  ✓ {name:25} {size:>12}  ({file_count} files)")
                else:
                    print(f"  ✓ {name:25} {size:>12}")
            else:
                print(f"  ✗ {name:25} {'Not cached':>12}")
    
    print("\n" + "=" * 60)


def clear_cache(project_dir: Path, cache_type: str):
    """Clear specific cache type."""
    data_dir = project_dir / "data"
    scripts_dir = project_dir / "scripts"
    
    cache_maps = {
        "raw": [
            data_dir / "raw" / "graph.graphml",
            data_dir / "raw" / "buildings.geojson",
            data_dir / "raw" / "landuse.geojson",
            data_dir / "raw" / "transit.geojson",
            data_dir / "raw" / "nodes.parquet",
            data_dir / "raw" / "edges.parquet",
        ],
        "pois": [
            data_dir / "raw" / "pois.geojson",
        ],
        "processed": [
            data_dir / "processed" / "graph.graphml",
            data_dir / "processed" / "nodes.parquet",
            data_dir / "processed" / "edges.parquet",
            data_dir / "processed" / "poi_node_mapping.parquet",
        ],
        "analysis": [
            data_dir / "analysis" / "nodes_with_scores.parquet",
            data_dir / "analysis" / "nodes_with_scores.csv",
            data_dir / "analysis" / "h3_agg.parquet",
            data_dir / "analysis" / "h3_agg.csv",
            data_dir / "analysis" / "metrics_summary.json",
        ],
        "viz": [
            scripts_dir / "interactive_map.html",
        ],
        "amenities": [
            scripts_dir / "cache" / "bengaluru_amenities",
            scripts_dir / "cache" / "bengaluru_areas_list.json",
        ],
        "osmnx": [
            scripts_dir / "cache",
        ],
        "all": [],  # Will be populated below
    }
    
    # "all" includes everything except OSMnx cache
    cache_maps["all"] = [
        item for key in ["raw", "pois", "processed", "analysis", "viz", "amenities"]
        for item in cache_maps[key]
    ]
    
    if cache_type not in cache_maps:
        print(f"❌ Unknown cache type: {cache_type}")
        print(f"Available types: {', '.join(cache_maps.keys())}")
        return
    
    items = cache_maps[cache_type]
    if not items:
        print(f"⚠️  No items defined for cache type: {cache_type}")
        return
    
    print(f"\n🗑️  Clearing {cache_type} cache...")
    removed = 0
    
    for item in items:
        if item.exists():
            if item.is_dir():
                shutil.rmtree(item)
                print(f"  ✓ Removed directory: {item}")
            else:
                item.unlink()
                print(f"  ✓ Removed file: {item}")
            removed += 1
    
    if removed == 0:
        print("  ℹ️  No cached files found to remove")
    else:
        print(f"\n✅ Cleared {removed} item(s)")


def main():
    parser = argparse.ArgumentParser(description="Manage PathLens cache")
    parser.add_argument("action", choices=["status", "clear"], help="Action to perform")
    parser.add_argument(
        "--type",
        choices=["raw", "pois", "processed", "analysis", "viz", "amenities", "osmnx", "all"],
        help="Type of cache to clear (required for 'clear' action)"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent  # Go up from pipeline/ to project root
    
    if args.action == "status":
        show_cache_status(project_dir)
    elif args.action == "clear":
        if not args.type:
            parser.error("--type is required for 'clear' action")
        
        print("\n⚠️  WARNING: This will delete cached data!")
        response = input(f"Clear '{args.type}' cache? [y/N]: ")
        
        if response.lower() in ['y', 'yes']:
            clear_cache(project_dir, args.type)
        else:
            print("Cancelled.")


if __name__ == "__main__":
    main()
