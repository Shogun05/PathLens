#!/usr/bin/env python3
"""Generate an interactive Leaflet preview of the street network and POIs."""
import argparse
from collections import defaultdict
from pathlib import Path

import folium
import geopandas as gpd
import osmnx as ox
import pandas as pd
from branca.colormap import linear
from shapely.geometry import LineString
from tqdm import tqdm

DEFAULT_RAW_DIR = Path("../data/raw/osm")
DEFAULT_PROCESSED_DIR = Path("../data/processed")


def load_data(graph_path: Path, pois_path: Path, nodes_path: Path, mapping_path: Path):
    G = ox.load_graphml(graph_path)
    pois = gpd.read_file(pois_path)
    nodes = gpd.read_parquet(nodes_path)
    mapping = pd.read_parquet(mapping_path)
    return G, pois, nodes, mapping


def attach_mapping(pois: gpd.GeoDataFrame, mapping: pd.DataFrame) -> gpd.GeoDataFrame:
    mapping = mapping.copy()
    if "poi_index" in mapping.index.names or mapping.index.name == "poi_index":
        mapping.index = mapping.index.astype(pois.index.dtype, copy=False)
    elif "poi_index" in mapping.columns:
        mapping.set_index("poi_index", inplace=True)
        mapping.index = mapping.index.astype(pois.index.dtype, copy=False)
    
    # Drop 'source' from mapping if it conflicts with pois
    if 'source' in mapping.columns and 'source' in pois.columns:
        mapping = mapping.drop(columns=['source'])
    
    enriched = pois.join(mapping, how="left")
    return enriched


def build_color_lookup(categories: pd.Series) -> dict:
    unique_categories = [cat for cat in categories.dropna().unique() if cat]
    if not unique_categories:
        return {}
    palette = linear.Set1_09.scale(0, max(len(unique_categories) - 1, 1))
    color_map = {}
    for idx, cat in enumerate(sorted(unique_categories)):
        color_map[cat] = palette(idx)
    return color_map


def add_poi_layers(m: folium.Map, pois: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, color_lookup: dict, show_links: bool):
    grouped = defaultdict(list)
    print("Grouping POIs by category...")
    for _, row in tqdm(pois.iterrows(), total=len(pois), desc="Grouping POIs", unit="poi"):
        cat = row.get("category_value") or "Other"
        grouped[cat].append(row)

    print(f"Creating {len(grouped)} POI layers...")
    for category, records in tqdm(grouped.items(), desc="Adding POI layers", unit="category"):
        layer = folium.FeatureGroup(name=f"POIs: {category}", show=True)
        color = color_lookup.get(category, "#FF8C00")
        for record in records:
            geom = record.geometry
            if geom is None or geom.is_empty:
                continue
            point_geom = geom if geom.geom_type == "Point" else geom.representative_point()
            node_id = record.get("nearest_node")
            popup_parts = [
                f"<b>POI:</b> {record.get('poi_id', record.name)}",
                f"<b>Category:</b> {category}",
            ]
            if pd.notna(record.get("distance_m")):
                popup_parts.append(f"<b>Distance:</b> {record['distance_m']:.1f} m")
            if pd.notna(node_id):
                popup_parts.append(f"<b>Nearest node:</b> {node_id}")
            popup = folium.Popup("<br/>".join(popup_parts), max_width=300)

            folium.CircleMarker(
                location=[point_geom.y, point_geom.x],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                weight=1,
                popup=popup,
            ).add_to(layer)

            if show_links and pd.notna(node_id) and node_id in nodes.index:
                node_geom = nodes.loc[node_id].geometry
                if node_geom is not None:
                    segment = LineString([(point_geom.x, point_geom.y), (node_geom.x, node_geom.y)])
                    folium.GeoJson(
                        segment,
                        name="poi-link",
                        style_function=lambda _:
                            {"color": color, "weight": 1.2, "opacity": 0.6, "dashArray": "4 4"},
                    ).add_to(layer)

        layer.add_to(m)


def add_node_heat_layer(m: folium.Map, nodes: gpd.GeoDataFrame):
    if "walkability" not in nodes.columns:
        return
    nodes_latlon = nodes.to_crs(epsg=4326) if nodes.crs and nodes.crs.to_epsg() != 4326 else nodes
    value_range = nodes_latlon["walkability"].replace([pd.NA, float("inf"), float("-inf")], pd.NA).dropna()
    if value_range.empty:
        return
    colormap = linear.YlGn_09.scale(value_range.min(), value_range.max())
    colormap.caption = "Walkability score"

    node_layer = folium.FeatureGroup(name="Nodes: walkability", show=False)
    for _, row in nodes_latlon.iterrows():
        value = row.get("walkability")
        if pd.isna(value):
            continue
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            color=colormap(value),
            fill=True,
            fill_color=colormap(value),
            fill_opacity=0.7,
            weight=0,
        ).add_to(node_layer)

    node_layer.add_to(m)
    colormap.add_to(m)


def make_map(
    graph_path: Path,
    pois_path: Path,
    nodes_path: Path,
    mapping_path: Path,
    output_path: Path,
    show_links: bool,
):
    print("Loading data for visualisation...")
    G, pois_raw, nodes, mapping = load_data(graph_path, pois_path, nodes_path, mapping_path)

    pois = attach_mapping(pois_raw, mapping)
    if pois.empty:
        raise ValueError("No POIs found for visualisation.")

    base_map = ox.plot_graph_folium(
        G,
        node_size=0,
        edge_width=1.2,
        edge_color="#4a4a4a",
        tiles="cartodbpositron",
    )

    color_lookup = build_color_lookup(pois.get("category_value", pd.Series(dtype=str)))
    add_poi_layers(base_map, pois, nodes, color_lookup, show_links)
    add_node_heat_layer(base_map, nodes)

    folium.TileLayer("CartoDB dark_matter", control=True, name="Dark").add_to(base_map)
    folium.LayerControl(collapsed=False).add_to(base_map)

    base_map.save(str(output_path))
    print(f"Interactive map saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create an interactive map for PathLens outputs.")
    parser.add_argument("--graph-path", default=DEFAULT_RAW_DIR / "graph.graphml", type=Path)
    parser.add_argument("--pois-path", default=DEFAULT_RAW_DIR / "pois.geojson", type=Path)
    parser.add_argument("--nodes-path", default=DEFAULT_PROCESSED_DIR / "nodes.parquet", type=Path)
    parser.add_argument(
        "--mapping-path",
        default=DEFAULT_PROCESSED_DIR / "poi_node_mapping.parquet",
        type=Path,
    )
    parser.add_argument("--out", default=Path("interactive_map.html"), type=Path)
    parser.add_argument("--no-links", action="store_true", help="Disable POI-to-node guide lines.")
    args = parser.parse_args()

    missing = [path for path in (args.graph_path, args.pois_path, args.nodes_path, args.mapping_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(str(p) for p in missing)}")

    make_map(
        graph_path=args.graph_path,
        pois_path=args.pois_path,
        nodes_path=args.nodes_path,
        mapping_path=args.mapping_path,
        output_path=args.out,
        show_links=not args.no_links,
    )


if __name__ == "__main__":
    main()
