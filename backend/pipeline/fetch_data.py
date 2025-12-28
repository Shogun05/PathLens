import osmnx as ox
import geopandas as gpd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

def fetch_map_data(location_name: str, output_dir: Path):
    """Downloads Graph and POIs for a location."""
    print(f"🌍 Fetching data for: {location_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download Graph
    try:
        G = ox.graph_from_place(location_name, network_type="walk", simplify=True)
        ox.save_graphml(G, output_dir / "graph.graphml")
    except Exception as e:
        raise ValueError(f"Could not find location '{location_name}'. {e}")

    # 2. Download POIs
    tags = {"amenity": ["school", "hospital"], "leisure": ["park"]}
    try:
        pois = ox.features_from_place(location_name, tags=tags)
        if not pois.empty:
            # Standardize to Points
            pois = pois.to_crs(epsg=32643) # Meters
            pois['geometry'] = pois.geometry.centroid
            pois = pois.to_crs(epsg=4326)  # Lat/Lon
            
            # Clean columns for GeoJSON
            for col in pois.columns:
                if pois[col].dtype == 'object':
                    pois[col] = pois[col].astype(str)
            
            pois.to_file(output_dir / "pois.geojson", driver="GeoJSON")
    except Exception:
        # Create empty if fail
        gpd.GeoDataFrame(columns=["amenity", "geometry"], geometry=[]).to_file(output_dir / "pois.geojson", driver="GeoJSON")
    
    return True