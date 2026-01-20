import pandas as pd
import os
import logging
import struct
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_wkb_point(wkb_bytes):
    """
    Parse a WKB Point (Little Endian) to (x, y).
    Format: 1 byte order, 4 bytes type, 8 bytes X, 8 bytes Y.
    Assumes Little Endian (1) and Point Type (1).
    """
    try:
        # Check byte order
        byte_order = wkb_bytes[0]
        # 1 = Little Endian, 0 = Big Endian
        endian = '<' if byte_order == 1 else '>'
        
        # Read X and Y (skip 5 bytes: 1 order + 4 type)
        x = struct.unpack(f'{endian}d', wkb_bytes[5:13])[0]
        y = struct.unpack(f'{endian}d', wkb_bytes[13:21])[0]
        return x, y
    except Exception:
        return None, None

def preprocess_city_pois(city_name: str, raw_parquet_path: str, output_path: str):
    """
    Preprocess raw POIs to filter and normalize amenities.
    Does NOT depend on geopandas/shapely.
    """
    logger.info(f"Preprocessing POIs for {city_name} from {raw_parquet_path}")
    
    if not os.path.exists(raw_parquet_path):
        logger.error(f"Raw POI file not found: {raw_parquet_path}")
        return False

    try:
        # Load raw data
        df = pd.read_parquet(raw_parquet_path)
        
        # Parse Geometry to Lat/Lon
        if 'geometry' in df.columns:
            # Apply parser
            coords = df['geometry'].apply(parse_wkb_point)
            # Create separate columns
            df['lon'] = [c[0] for c in coords]
            df['lat'] = [c[1] for c in coords]
        else:
            logger.error("No geometry column found")
            return False
            
        # Ensure 'amenity' column exists
        if 'amenity' not in df.columns:
            df['amenity'] = ''
        else:
            df['amenity'] = df['amenity'].fillna('')
            
        # Ensure 'leisure' column exists
        if 'leisure' not in df.columns:
            df['leisure'] = ''
        else:
            df['leisure'] = df['leisure'].fillna('')
            
        # Ensure 'highway' column exists
        if 'highway' not in df.columns:
            df['highway'] = ''
        else:
            df['highway'] = df['highway'].fillna('')

        # Create a new 'normalized_type' column
        df['amenity_type'] = None

        # 1. Hospitals / Healthcare
        healthcare_types = ['hospital', 'clinic', 'doctors', 'nursing_home']
        mask_healthcare = df['amenity'].isin(healthcare_types)
        df.loc[mask_healthcare, 'amenity_type'] = 'hospital'
        
        # 2. Schools / Education
        education_types = ['school', 'kindergarten', 'university', 'college']
        mask_education = df['amenity'].isin(education_types)
        df.loc[mask_education, 'amenity_type'] = 'school'
        
        # 3. Parks / Leisure
        park_types = ['park', 'garden', 'nature_reserve', 'playground']
        mask_park = df['leisure'].isin(park_types)
        df.loc[mask_park, 'amenity_type'] = 'park'
        
        # 4. Bus Stations / Stops
        # Check amenity=bus_station OR highway=bus_stop
        mask_bus = (df['amenity'] == 'bus_station') | (df['highway'] == 'bus_stop')
        df.loc[mask_bus, 'amenity_type'] = 'bus'

        # Filter to only rows with a normalized type AND valid coordinates
        processed_df = df[
            (df['amenity_type'].notna()) & 
            (df['lon'].notna()) & 
            (df['lat'].notna())
        ].copy()
        
        # Keep identifying info and new coords
        # Dropping geometry to save space and avoid WKB issues downstream
        keep_cols = ['element_type', 'osmid', 'amenity_type', 'lat', 'lon', 'name']
        processed_df = processed_df[[c for c in keep_cols if c in processed_df.columns]]
        
        logger.info(f"Filtered {len(df)} POIs down to {len(processed_df)} relevant amenities for {city_name}")
        
        # Convert to GeoJSON features
        features = []
        total_rows = len(processed_df)
        
        logger.info(f"Converting {total_rows} POIs to GeoJSON format for {city_name}...")
        
        for idx, row in processed_df.iterrows():
            if (idx + 1) % 1000 == 0:
                 logger.info(f"Processed {idx + 1}/{total_rows} POIs...")
                 
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['lon'], row['lat']]
                },
                "properties": {
                    "id": str(row['osmid']),
                    "amenity_type": row['amenity_type'],
                    "name": row['name']
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(geojson, f)
            
        logger.info(f"Saved processed POIs to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to preprocess POIs for {city_name}: {e}")
        return False

if __name__ == "__main__":
    # Test run for Bangalore
    base_dir = Path(__file__).parent.parent
    city = "bengaluru"  # normalized name
    raw_path = base_dir / "data" / "cities" / "bangalore" / "raw" / "osm" / "pois.parquet"
    out_path = base_dir / "data" / "cities" / "bangalore" / "graph" / "processed_pois.json"
    
    # Ensure graph dir exists
    os.makedirs(out_path.parent, exist_ok=True)
    
    if raw_path.exists():
        preprocess_city_pois(city, str(raw_path), str(out_path))
    else:
        print(f"File not found for testing: {raw_path}")
