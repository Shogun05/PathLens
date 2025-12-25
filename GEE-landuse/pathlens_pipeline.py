#!/usr/bin/env python3
"""
PathLens Complete Pipeline - Single File
Integrates: amenity placement → GEE upload → GEE analysis → local download → feasibility filtering

Usage:
    python pathlens_pipeline.py hospital
    python pathlens_pipeline.py school
    python pathlens_pipeline.py --all  # Process all amenities
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import ee
import geemap


# ================= CONFIGURATION =================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ID = 'tidecon-9913b'

# Input files
BEST_CANDIDATE_PATH = BASE_DIR / "best_candidate.json"
NODES_CSV_PATH = BASE_DIR / "optimized_nodes_with_scores.csv"

# Output directory
OUTPUT_DIR = BASE_DIR / "gee_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Minimum required free-land area per amenity type (m²)
MIN_AREA = {
    'hospital': 1000,
    'school': 800,
    'park': 500,
    'pharmacy': 100,
    'supermarket': 600,
    'bus_station': 400
}

BUFFER_METERS = 200


# ================= STEP 1: AMENITY PLACEMENT =================

class AmenityPlacementBuilder:
    """Builds GEE candidate points from GA results"""
    
    def __init__(self, best_candidate_path, nodes_csv_path):
        self.best_candidate_path = best_candidate_path
        self.nodes_csv_path = nodes_csv_path
    
    def parse_best_candidate(self) -> dict:
        """Return dict: amenity -> list[node_ids] from best_candidate.json."""
        with self.best_candidate_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        cand_str = data["candidate"]
        amenity_map = {}
        
        # Example: bus_station:10020025539,10991934818|hospital:10035832527,...
        for block in cand_str.split("|"):
            if not block.strip():
                continue
            name, ids_str = block.split(":")
            node_ids = [int(x) for x in ids_str.split(",") if x]
            amenity_map[name.strip()] = node_ids
        
        return amenity_map
    
    def create_geojson(self, amenity: str) -> Path:
        """Create GeoJSON file for specific amenity"""
        print(f"\n{'='*60}")
        print(f"STEP 1: Creating GeoJSON for {amenity.upper()}")
        print(f"{'='*60}")
        
        # Parse GA output
        amenity_map = self.parse_best_candidate()
        if amenity not in amenity_map:
            raise ValueError(
                f"Amenity '{amenity}' not found in best_candidate.json. "
                f"Available: {list(amenity_map.keys())}"
            )
        candidate_ids = set(amenity_map[amenity])
        
        # Load node table
        df = pd.read_csv(self.nodes_csv_path)
        
        id_col = "osmid"
        lat_col = "lat"
        lon_col = "lon"
        
        # Filter nodes
        df_sub = df[df[id_col].isin(candidate_ids)].copy()
        if df_sub.empty:
            raise RuntimeError(f"No nodes found for amenity '{amenity}'")
        
        # Build GeoDataFrame
        df_sub["node_id"] = df_sub[id_col].astype("int64")
        df_sub["amenity"] = amenity
        
        geometry = [Point(xy) for xy in zip(df_sub[lon_col], df_sub[lat_col])]
        gdf = gpd.GeoDataFrame(
            df_sub[["node_id", "amenity"]],
            geometry=geometry,
            crs="EPSG:4326",
        )
        
        # Save GeoJSON
        out_path = OUTPUT_DIR / f"node_candidates_{amenity}.geojson"
        gdf.to_file(out_path, driver="GeoJSON")
        
        print(f"✓ Created {len(gdf)} candidate nodes → {out_path.name}")
        return out_path


# ================= STEP 2: UPLOAD TO GEE =================

class GEEUploader:
    """Upload GeoJSON to Google Earth Engine"""
    
    def __init__(self, project_id):
        self.project_id = project_id
        self._initialize_ee()
    
    def _initialize_ee(self):
        """Initialize Earth Engine"""
        try:
            ee.Initialize()
        except:
            ee.Authenticate()
            ee.Initialize()
    
    def upload_geojson(self, geojson_path: Path, amenity: str) -> str:
        """Upload GeoJSON to GEE and return asset ID"""
        print(f"\n{'='*60}")
        print(f"STEP 2: Uploading {amenity.upper()} to GEE")
        print(f"{'='*60}")
        
        asset_id = f'projects/{self.project_id}/assets/{amenity}_nodes'
        
        # Read GeoJSON
        gdf = gpd.read_file(geojson_path)
        print(f"✓ Loaded {len(gdf)} nodes from {geojson_path.name}")
        
        # Convert to Earth Engine FeatureCollection
        features = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            props = {}
            
            for col in gdf.columns:
                if col != 'geometry':
                    value = row[col]
                    if pd.notna(value):
                        if isinstance(value, (np.integer, np.int64)):
                            props[col] = int(value)
                        elif isinstance(value, (np.floating, float)):
                            props[col] = float(value)
                        else:
                            props[col] = str(value)
            
            geom_json = geom.__geo_interface__
            ee_geom = ee.Geometry(geom_json)
            feature = ee.Feature(ee_geom, props)
            features.append(feature)
        
        fc = ee.FeatureCollection(features)
        
        # Export to asset
        print(f"✓ Starting upload to {asset_id}...")
        task = ee.batch.Export.table.toAsset(
            collection=fc,
            description=f'{amenity}_nodes_upload',
            assetId=asset_id
        )
        task.start()
        
        # Wait for completion
        print("✓ Waiting for upload to complete...", end="", flush=True)
        while task.status()['state'] in ['READY', 'RUNNING']:
            print(".", end="", flush=True)
            time.sleep(5)
        
        status = task.status()
        if status['state'] == 'COMPLETED':
            print(" ✓ COMPLETED")
            return asset_id
        else:
            raise RuntimeError(f"Upload failed: {status}")


# ================= STEP 3: GEE ANALYSIS =================

class PathLensAnalyzer:
    """Run land-use feasibility analysis on GEE"""
    
    def __init__(self, project_id):
        self.project_id = project_id
        self.MIN_AREA = MIN_AREA
        self.BUFFER_METERS = BUFFER_METERS
        self._initialize_ee()
        self._load_earth_engine_data()
    
    def _initialize_ee(self):
        try:
            ee.Initialize()
        except:
            ee.Authenticate()
            ee.Initialize()
    
    def _load_earth_engine_data(self):
        """Load ESA WorldCover data"""
        self.worldCover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
        self.freeMask = (self.worldCover.eq(30)
                         .Or(self.worldCover.eq(40))
                         .Or(self.worldCover.eq(60)))
        self.pixelArea = ee.Image.pixelArea()
    
    def process_node(self, feature):
        """Process a single node to find feasible placement areas"""
        f = ee.Feature(feature)
        amenity = ee.String(f.get('amenity'))
        
        bufferGeom = f.geometry().buffer(self.BUFFER_METERS, 1)
        freeInBuffer = self.freeMask.clip(bufferGeom)
        
        areaDict = (self.pixelArea
                    .updateMask(freeInBuffer)
                    .reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=bufferGeom,
                        scale=10,
                        maxPixels=1e9
                    ))
        
        freeAreaM2 = ee.Number(areaDict.get('area')).round()
        minReq = ee.Number(ee.Dictionary(self.MIN_AREA).get(amenity))
        feasible = freeAreaM2.gte(minReq)
        
        emptyResult = f.set({
            'free_area_m2': freeAreaM2,
            'min_area_req': minReq,
            'feasible': feasible,
            'has_patch': False
        })
        
        def get_patches():
            return freeInBuffer.reduceToVectors(
                geometry=bufferGeom,
                scale=10,
                geometryType='polygon',
                eightConnected=True,
                labelProperty='label',
                maxPixels=1e8
            )
        
        polyFC = ee.Algorithms.If(feasible, get_patches(), ee.FeatureCollection([]))
        patches = ee.FeatureCollection(polyFC).map(lambda p: self._add_patch_info(p, f))
        
        patchCount = patches.size()
        filteredPatches = patches.sort('distance_m')
        bestPatchList = filteredPatches.limit(1).toList(1)
        hasPatch = bestPatchList.size().gt(0)
        
        bestPatch = ee.Feature(
            ee.Algorithms.If(hasPatch, bestPatchList.get(0), ee.Feature(None))
        )
        
        return emptyResult.set({
            'has_patch': hasPatch,
            'best_patch': bestPatch,
            'patch_count': patchCount
        })
    
    def _add_patch_info(self, patch, node):
        p = ee.Feature(patch)
        geom = p.geometry()
        area = geom.area(1)
        centroid = geom.centroid(1)
        dist = centroid.distance(node.geometry())
        
        return p.set({
            'area_m2': area,
            'distance_m': dist
        })
    
    def _create_placement(self, feature):
        f = ee.Feature(feature)
        patch = ee.Feature(f.get('best_patch'))
        
        return patch.set({
            'node_id': f.get('node_id'),
            'amenity': f.get('amenity'),
            'free_area_m2': f.get('free_area_m2'),
            'min_area_req': f.get('min_area_req'),
            'distance_m': patch.get('distance_m'),
            'patch_area_m2': patch.get('area_m2')
        })
    
    def analyze_amenity(self, amenity: str, asset_id: str):
        """Run GEE analysis and return results"""
        print(f"\n{'='*60}")
        print(f"STEP 3: Running GEE Analysis for {amenity.upper()}")
        print(f"{'='*60}")
        
        nodes = ee.FeatureCollection(asset_id)
        node_count = nodes.size().getInfo()
        print(f"✓ Loaded {node_count} nodes from GEE")
        
        print(f"✓ Processing nodes with {self.BUFFER_METERS}m buffer...")
        results = nodes.map(self.process_node)
        
        # Create feasibility table
        feasibility = results.map(lambda f: ee.Feature(None, {
            'node_id': f.get('node_id'),
            'amenity': f.get('amenity'),
            'free_area_m2': f.get('free_area_m2'),
            'min_area_req': f.get('min_area_req'),
            'feasible': f.get('feasible'),
            'has_patch': f.get('has_patch'),
            'patch_count': f.get('patch_count')
        }))
        
        # Create placement polygons
        placements = (results
                      .filter(ee.Filter.eq('feasible', 1))
                      .filter(ee.Filter.eq('has_patch', 1))
                      .map(self._create_placement))
        
        placements = ee.FeatureCollection(placements)
        
        # Get statistics
        feasible_count = feasibility.filter(ee.Filter.eq('feasible', 1)).size().getInfo()
        placement_count = placements.size().getInfo()
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS RESULTS:")
        print(f"  Total nodes: {node_count}")
        print(f"  Feasible nodes: {feasible_count}")
        print(f"  Nodes with placements: {placement_count}")
        print(f"{'='*60}")
        
        return {
            'feasibility': feasibility,
            'placements': placements,
            'stats': {
                'total': node_count,
                'feasible': feasible_count,
                'with_placements': placement_count
            }
        }


# ================= STEP 4: DOWNLOAD TO LOCAL =================

class GEEDownloader:
    """Download GEE results to local files"""
    
    def download_to_local(self, feasibility_fc, placements_fc, amenity: str, output_dir: Path):
        """Download feasibility CSV and placements GeoJSON to local directory"""
        print(f"\n{'='*60}")
        print(f"STEP 4: Downloading Results for {amenity.upper()}")
        print(f"{'='*60}")
        
        # Download feasibility as CSV
        feas_output = output_dir / f"pathlens_feasibility_{amenity}.csv"
        print(f"✓ Downloading feasibility data...", end="", flush=True)
        
        # Get data as list of features
        feas_data = feasibility_fc.getInfo()['features']
        feas_rows = [f['properties'] for f in feas_data]
        feas_df = pd.DataFrame(feas_rows)
        feas_df.to_csv(feas_output, index=False)
        print(f" → {feas_output.name}")
        
        # Download placements as GeoJSON
        place_output = output_dir / f"pathlens_placements_{amenity}.geojson"
        print(f"✓ Downloading placement polygons...", end="", flush=True)
        
        geemap.ee_export_geojson(placements_fc, filename=str(place_output))
        print(f" → {place_output.name}")
        
        return feas_output, place_output


# ================= STEP 5: FEASIBILITY FILTER =================

class FeasibilityFilter:
    """Filter and merge feasibility results with node attributes"""
    
    def __init__(self, nodes_csv_path):
        self.nodes_csv_path = nodes_csv_path
    
    def filter_and_merge(self, amenity: str, feas_csv: Path, placements_geojson: Path, output_dir: Path):
        """Filter feasible nodes and merge with node attributes"""
        print(f"\n{'='*60}")
        print(f"STEP 5: Filtering Feasible Nodes for {amenity.upper()}")
        print(f"{'='*60}")
        
        # Load data
        nodes = pd.read_csv(self.nodes_csv_path)
        feas = pd.read_csv(feas_csv)
        placements = gpd.read_file(placements_geojson)
        
        print(f"✓ Loaded {len(nodes)} nodes from CSV")
        print(f"✓ Loaded {len(feas)} feasibility records")
        print(f"✓ Loaded {len(placements)} placement polygons")
        
        # Filter feasibility to this amenity
        feas_amen = feas[feas["amenity"] == amenity].copy()
        
        if feas_amen["feasible"].dtype != bool:
            feas_amen["feasible"] = feas_amen["feasible"].astype(bool)
        
        feasible_only = feas_amen[feas_amen["feasible"]]
        print(f"✓ Found {len(feasible_only)} feasible nodes")
        
        # Merge with node table
        nodes = nodes.copy()
        nodes["osmid"] = nodes["osmid"].astype("int64")
        feasible_only["node_id"] = feasible_only["node_id"].astype("int64")
        
        merged = nodes.merge(
            feasible_only,
            left_on="osmid",
            right_on="node_id",
            how="inner",
            suffixes=("", "_gee")
        )
        
        # Clean placements
        if placements.crs is None:
            placements.set_crs(epsg=4326, inplace=True)
        
        placements_clean = placements[placements["amenity"] == amenity]
        
        # Save outputs
        out_feas_nodes = output_dir / f"gee_feasible_nodes_{amenity}.csv"
        out_merged = output_dir / f"gee_feasible_nodes_{amenity}_merged.csv"
        out_placements = output_dir / f"gee_placements_{amenity}.geojson"
        
        feasible_only.to_csv(out_feas_nodes, index=False)
        merged.to_csv(out_merged, index=False)
        placements_clean.to_file(out_placements, driver="GeoJSON")
        
        print(f"\n✓ Saved feasible nodes → {out_feas_nodes.name}")
        print(f"✓ Saved merged data → {out_merged.name}")
        print(f"✓ Saved placements → {out_placements.name}")
        
        # Show sample
        print(f"\nTop 5 feasible nodes:")
        cols_show = ['osmid', 'lon', 'lat', 'free_area_m2', 'min_area_req', 'feasible']
        if 'walkability' in merged.columns:
            cols_show.append('walkability')
        if 'accessibility_score' in merged.columns:
            cols_show.append('accessibility_score')
        cols_show = [c for c in cols_show if c in merged.columns]
        print(merged[cols_show].head().to_string(index=False))
        
        return out_feas_nodes, out_merged, out_placements


# ================= MAIN PIPELINE =================

class PathLensPipeline:
    """Complete PathLens pipeline orchestrator"""
    
    def __init__(self):
        self.placement_builder = AmenityPlacementBuilder(BEST_CANDIDATE_PATH, NODES_CSV_PATH)
        self.gee_uploader = GEEUploader(PROJECT_ID)
        self.analyzer = PathLensAnalyzer(PROJECT_ID)
        self.downloader = GEEDownloader()
        self.filter = FeasibilityFilter(NODES_CSV_PATH)
    
    def process_amenity(self, amenity: str):
        """Run complete pipeline for a single amenity"""
        start_time = time.time()
        
        print(f"\n{'#'*60}")
        print(f"# PATHLENS PIPELINE: {amenity.upper()}")
        print(f"{'#'*60}")
        
        try:
            # Step 1: Create GeoJSON
            geojson_path = self.placement_builder.create_geojson(amenity)
            
            # Step 2: Upload to GEE
            asset_id = self.gee_uploader.upload_geojson(geojson_path, amenity)
            
            # Step 3: Run GEE analysis
            results = self.analyzer.analyze_amenity(amenity, asset_id)
            
            # Step 4: Download results locally
            feas_csv, place_geojson = self.downloader.download_to_local(
                results['feasibility'],
                results['placements'],
                amenity,
                OUTPUT_DIR
            )
            
            # Step 5: Filter and merge
            self.filter.filter_and_merge(amenity, feas_csv, place_geojson, OUTPUT_DIR)
            
            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"✓ PIPELINE COMPLETED for {amenity.upper()} in {elapsed:.1f}s")
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            print(f"\n✗ ERROR processing {amenity}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_all_amenities(self):
        """Run pipeline for all amenities"""
        amenities = list(MIN_AREA.keys())
        
        print(f"\n{'#'*60}")
        print(f"# PATHLENS BATCH PIPELINE: {len(amenities)} AMENITIES")
        print(f"{'#'*60}")
        
        results = {}
        for amenity in amenities:
            success = self.process_amenity(amenity)
            results[amenity] = success
        
        # Summary
        print(f"\n{'='*60}")
        print("BATCH PIPELINE SUMMARY")
        print(f"{'='*60}")
        for amenity, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"  {amenity:15} | {status}")
        
        return results


# ================= ENTRY POINT =================

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pathlens_pipeline.py <amenity>     # Process single amenity")
        print("  python pathlens_pipeline.py --all         # Process all amenities")
        print("\nAvailable amenities:", list(MIN_AREA.keys()))
        sys.exit(1)
    
    pipeline = PathLensPipeline()
    
    if sys.argv[1] == '--all':
        pipeline.process_all_amenities()
    else:
        amenity = sys.argv[1]
        if amenity not in MIN_AREA:
            print(f"Error: Unknown amenity '{amenity}'")
            print(f"Available: {list(MIN_AREA.keys())}")
            sys.exit(1)
        pipeline.process_amenity(amenity)


if __name__ == "__main__":
    main()