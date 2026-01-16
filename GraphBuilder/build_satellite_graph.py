#!/usr/bin/env python3
"""
Build OSM-compatible graph from satellite imagery.

This is the main script that:
1. Converts Sat2Graph tile outputs to geo-referenced graphs
2. Merges multiple tiles into a single unified graph
3. Outputs to data/raw/osm/ format for the pipeline

Usage:
    python build_satellite_graph.py --input-dir custom_outputs/ \
                                     --metadata downloader/metadata.json \
                                     --output-dir ../data/raw/osm/

After running, continue with the standard pipeline:
    cd data-pipeline
    python build_graph.py --graph-path ../data/raw/osm/graph.graphml \
                          --pois-path ../data/raw/osm/pois.parquet
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

# Import from local modules
from sat2graph_converter import (
    load_metadata,
    graph_to_gdfs,
    add_osmnx_metadata,
    haversine_distance
)
from merge_tile_graphs import merge_tile_graphs



def save_osm_compatible_outputs(
    G: nx.MultiDiGraph,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    output_dir: Path
) -> None:
    """
    Save graph in OSM-compatible format to output directory.
    
    Creates:
    - graph.graphml (NetworkX graph with all attributes as strings)
    - nodes.parquet (Node GeoDataFrame)
    - edges.parquet (Edge GeoDataFrame)
    
    Note: POIs, landuse, and transit files are NOT created.
    Use existing OSM data for these files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Saving OSM-compatible outputs to {output_dir}")
    
    # ----- GraphML -----
    # OSMnx-compatible GraphML format
    graph_path = output_dir / "graph.graphml"
    
    # Convert all attributes to strings for OSMnx compatibility
    G_export = G.copy()
    
    # Convert node attributes to strings
    for node_id in G_export.nodes():
        node_data = G_export.nodes[node_id]
        for key, value in list(node_data.items()):
            if key == 'geometry':
                # Convert geometry to WKT string
                if hasattr(value, 'wkt'):
                    node_data[key] = value.wkt
            else:
                # Convert all other attributes to strings
                node_data[key] = str(value)
    
    # Convert edge attributes to strings
    for u, v, k in G_export.edges(keys=True):
        edge_data = G_export.edges[u, v, k]
        for key, value in list(edge_data.items()):
            if key == 'geometry':
                # Convert geometry to WKT string
                if hasattr(value, 'wkt'):
                    edge_data[key] = value.wkt
            else:
                # Convert all other attributes to strings
                edge_data[key] = str(value)
    
    nx.write_graphml(G_export, graph_path)
    print(f"   ✅ graph.graphml: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # ----- Nodes Parquet -----
    nodes_gdf.to_parquet(output_dir / "nodes.parquet")
    print(f"   ✅ nodes.parquet: {len(nodes_gdf)} rows")
    
    # ----- Edges Parquet -----
    edges_gdf.to_parquet(output_dir / "edges.parquet")
    print(f"   ✅ edges.parquet: {len(edges_gdf)} rows")


def print_summary(G: nx.MultiDiGraph) -> None:
    """Print summary statistics of the converted graph."""
    print("\n" + "="*60)
    print("📊 CONVERSION SUMMARY")
    print("="*60)
    
    print(f"\n🔢 Graph Statistics:")
    print(f"   Nodes: {G.number_of_nodes():,}")
    print(f"   Edges: {G.number_of_edges():,}")
    
    if G.number_of_nodes() > 0:
        # Geographic bounds
        lats = [data['y'] for _, data in G.nodes(data=True)]
        lons = [data['x'] for _, data in G.nodes(data=True)]
        print(f"\n🌍 Geographic Bounds:")
        print(f"   Latitude:  {min(lats):.6f} to {max(lats):.6f}")
        print(f"   Longitude: {min(lons):.6f} to {max(lons):.6f}")
        
        # Edge length statistics
        lengths = [data.get('length', 0) for _, _, data in G.edges(data=True)]
        if lengths:
            print(f"\n📏 Edge Lengths:")
            print(f"   Min:    {min(lengths):.1f} m")
            print(f"   Max:    {max(lengths):.1f} m")
            print(f"   Mean:   {sum(lengths)/len(lengths):.1f} m")
            print(f"   Total:  {sum(lengths)/1000:.1f} km")
        
        # Node degree statistics
        degrees = [d for _, d in G.degree()]
        print(f"\n🔗 Node Degrees:")
        print(f"   Min: {min(degrees)}")
        print(f"   Max: {max(degrees)}")
        print(f"   Mean: {sum(degrees)/len(degrees):.1f}")
        
        # Count by degree
        dead_ends = sum(1 for d in degrees if d <= 2)
        intersections = sum(1 for d in degrees if d > 2)
        print(f"   Dead ends/midblocks: {dead_ends:,}")
        print(f"   Intersections (3+): {intersections:,}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Build OSM-compatible graph from satellite imagery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single tile
  python build_satellite_graph.py -i custom_outputs/ -m downloader/metadata.json

  # Convert and save to data/raw/osm/
  python build_satellite_graph.py -i custom_outputs/ -m downloader/metadata.json -o ../data/raw/osm/

  # With custom snap threshold
  python build_satellite_graph.py -i custom_outputs/ -m downloader/metadata.json -s 15

After conversion, continue with:
  cd ../data-pipeline
  python build_graph.py --graph-path ../data/raw/osm/graph.graphml --pois-path ../data/raw/osm/pois.parquet
        """
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Directory containing *_graph.p files from Sat2Graph"
    )
    parser.add_argument(
        "--metadata", "-m",
        type=Path,
        required=True,
        help="Path to metadata.json from downloader"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: ../data/raw/osm/)"
    )
    parser.add_argument(
        "--snap-threshold", "-s",
        type=float,
        default=10.0,
        help="Snap threshold in meters for merging boundary nodes (default: 10)"
    )
    parser.add_argument(
        "--keep-existing-pois",
        type=Path,
        default=None,
        help="Path to existing pois.parquet to copy instead of creating empty"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        # Default to ../data/raw/osm/ relative to GraphBuilder
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir.parent / "data" / "raw" / "osm"
    else:
        output_dir = args.output_dir
    
    print("="*60)
    print("🛰️  SATELLITE GRAPH BUILDER")
    print("="*60)
    print(f"\n📂 Input directory: {args.input_dir}")
    print(f"📋 Metadata file:   {args.metadata}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📏 Snap threshold:  {args.snap_threshold}m")
    
    # Check inputs exist
    if not args.input_dir.exists():
        print(f"\n❌ Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    if not args.metadata.exists():
        print(f"\n❌ Error: Metadata file does not exist: {args.metadata}")
        sys.exit(1)
    
    # Merge tile graphs
    print("\n" + "-"*60)
    merged_G = merge_tile_graphs(
        args.input_dir,
        args.metadata,
        args.snap_threshold
    )
    
    if merged_G.number_of_nodes() == 0:
        print("\n❌ Error: No valid graph data found!")
        sys.exit(1)
    
    # Convert to GeoDataFrames
    print("\n🔄 Converting to GeoDataFrames...")
    nodes_gdf, edges_gdf = graph_to_gdfs(merged_G)
    
    # Save outputs
    save_osm_compatible_outputs(
        merged_G,
        nodes_gdf,
        edges_gdf,
        output_dir
    )
    
    # Copy existing POIs if specified
    if args.keep_existing_pois and args.keep_existing_pois.exists():
        import shutil
        dest = output_dir / "pois.parquet"
        shutil.copy(args.keep_existing_pois, dest)
        print(f"   ✅ Copied existing POIs from {args.keep_existing_pois}")
    
    # Print summary
    print_summary(merged_G)
    
    print("\n✅ Conversion complete!")
    print("\n📌 Next steps:")
    print("   1. Review the output files in the output directory")
    print("   2. Populate pois.parquet with amenity data (from OSM or other source)")
    print("   3. Run the data pipeline:")
    print(f"      cd data-pipeline")
    print(f"      python build_graph.py --graph-path {output_dir}/graph.graphml \\")
    print(f"                            --pois-path {output_dir}/pois.parquet")


if __name__ == "__main__":
    main()
