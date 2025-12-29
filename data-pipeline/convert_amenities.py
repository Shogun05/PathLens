#!/usr/bin/env python3
"""
Convert fetched Bengaluru amenity JSON files to a single GeoJSON file
compatible with the PathLens pipeline.
"""
import json
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from tqdm import tqdm


def load_amenity_files(cache_dir: Path):
    """Load all amenity JSON files from the cache directory."""
    amenities = []
    
    json_files = sorted(cache_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {cache_dir}")
    
    print(f"Loading {len(json_files)} area files...")
    
    for json_file in tqdm(json_files, desc="Loading area files", unit="file"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            area_name = data.get("area_name", json_file.stem)
            for amenity in data.get("amenities", []):
                amenity_copy = amenity.copy()
                amenity_copy["area_name"] = area_name
                amenities.append(amenity_copy)
        
        except Exception as e:
            tqdm.write(f"⚠️  Error loading {json_file.name}: {e}")
            continue
    
    print(f"✓ Loaded {len(amenities)} total amenities")
    return amenities


def convert_to_geodataframe(amenities):
    """Convert amenities list to GeoDataFrame."""
    records = []
    
    for amenity in tqdm(amenities, desc="Converting to GeoDataFrame", unit="amenity"):
        tags = amenity.get("tags", {})
        
        # Get coordinates based on element type
        if amenity["type"] == "node":
            lat = amenity.get("lat")
            lon = amenity.get("lon")
        elif amenity["type"] == "way" and "center" in amenity:
            lat = amenity["center"].get("lat")
            lon = amenity["center"].get("lon")
        elif amenity["type"] == "relation" and "center" in amenity:
            lat = amenity["center"].get("lat")
            lon = amenity["center"].get("lon")
        else:
            # Skip if no coordinates available
            continue
        
        if lat is None or lon is None:
            continue
        
        # Extract relevant tags
        record = {
            "osmid": amenity["id"],
            "element_type": amenity["type"],
            "geometry": Point(lon, lat),
            "area_name": amenity.get("area_name"),
            # Primary categorization
            "amenity": tags.get("amenity"),
            "shop": tags.get("shop"),
            "leisure": tags.get("leisure"),
            # Additional attributes
            "name": tags.get("name"),
            "name:en": tags.get("name:en"),
            "name:kn": tags.get("name:kn"),
            "addr:street": tags.get("addr:street"),
            "addr:city": tags.get("addr:city"),
            "addr:postcode": tags.get("addr:postcode"),
            "opening_hours": tags.get("opening_hours"),
            "phone": tags.get("phone"),
            "website": tags.get("website"),
            "cuisine": tags.get("cuisine"),
            "brand": tags.get("brand"),
        }
        
        records.append(record)
    
    if not records:
        raise ValueError("No valid amenity records to convert")
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    
    print(f"✓ Created GeoDataFrame with {len(gdf)} amenities")
    print(f"  - Amenities: {gdf['amenity'].notna().sum()}")
    print(f"  - Shops: {gdf['shop'].notna().sum()}")
    print(f"  - Leisure: {gdf['leisure'].notna().sum()}")
    
    return gdf


def save_geojson(gdf: gpd.GeoDataFrame, output_path: Path):
    """Save GeoDataFrame as GeoJSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"💾 Saved to: {output_path}")


def main():
    # Path adjustments for new structure (data-pipeline/ at project root)
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent  # Go up from data-pipeline/ to project root
    cache_dir = project_dir / "data" / "raw" / "bengaluru" / "bengaluru_amenities"
    output_path = project_dir / "data" / "raw" / "osm" / "pois.geojson"
    
    print("🔄 Converting Bengaluru amenities to GeoJSON...")
    print(f"📁 Source: {cache_dir}")
    print(f"📁 Output: {output_path}")
    print("=" * 60)
    
    # Load all amenity files
    amenities = load_amenity_files(cache_dir)
    
    # Convert to GeoDataFrame
    gdf = convert_to_geodataframe(amenities)
    
    # Save as GeoJSON
    save_geojson(gdf, output_path)
    
    print("\n✅ Conversion complete!")
    print(f"Total POIs: {len(gdf)}")
    print(f"Unique areas: {gdf['area_name'].nunique()}")


if __name__ == "__main__":
    main()
