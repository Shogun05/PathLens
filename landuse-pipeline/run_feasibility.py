#!/usr/bin/env python3
"""
Land-Use Feasibility Pipeline - Single File
Integrates: amenity placement → GEE upload → GEE analysis → local download → feasibility filtering

Usage:
    python run_feasibility.py hospital
    python run_feasibility.py school
    python run_feasibility.py --all  # Process all amenities
"""

import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import ee
import geemap
from google.oauth2 import service_account


# ================= LOGGING CONFIGURATION =================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ================= CONFIGURATION =================

# landuse-pipeline/ is at project root, so parent is BASE_DIR
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ID = 'tidecon-9913b'
SERVICE_ACCOUNT_JSON = Path(__file__).resolve().parent / "landuse-service-account.json"

# Input files
BEST_CANDIDATE_PATH = BASE_DIR / "data" / "optimization" / "runs" / "best_candidate.json"
NODES_PARQUET_PATH = BASE_DIR / "data" / "analysis" / "nodes_with_scores.parquet"

# Output directory
OUTPUT_DIR = BASE_DIR / "data" / "landuse"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum required free-land area per amenity type (m²)
MIN_AREA = {
    'hospital': 1000,
    'school': 800,
    'park': 500,
    'pharmacy': 100,
    'supermarket': 600,
    'bus_station': 400,
    'bank': 150
}

BUFFER_METERS = 200


# ================= STEP 1: AMENITY PLACEMENT =================

class AmenityPlacementBuilder:
    """Builds GEE candidate points from GA results"""
    
    def __init__(self, best_candidate_path, nodes_parquet_path):
        self.best_candidate_path = best_candidate_path
        self.nodes_parquet_path = nodes_parquet_path
    
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
        logger.info("="*60)
        logger.info(f"STEP 1: Creating GeoJSON for {amenity.upper()}")
        logger.info("="*60)
        
        # Parse GA output
        amenity_map = self.parse_best_candidate()
        if amenity not in amenity_map:
            raise ValueError(
                f"Amenity '{amenity}' not found in best_candidate.json. "
                f"Available: {list(amenity_map.keys())}"
            )
        candidate_ids = set(amenity_map[amenity])
        
        # Load node table from parquet - reset index to make osmid a column
        df = pd.read_parquet(self.nodes_parquet_path).reset_index()
        
        # Verify osmid column exists
        if 'osmid' not in df.columns:
            raise ValueError(f"'osmid' column not found. Available columns: {list(df.columns)}")
        
        # Filter nodes
        df_sub = df[df['osmid'].isin(candidate_ids)].copy()
        if df_sub.empty:
            raise RuntimeError(f"No nodes found for amenity '{amenity}'")
        
        # Build GeoDataFrame from lon/lat coordinates
        df_sub["node_id"] = df_sub['osmid'].astype("int64")
        df_sub["amenity"] = amenity
        
        # Create Point geometries from lon/lat columns
        # The parquet file has lon/lat in WGS84 (EPSG:4326)
        geometry = [Point(xy) for xy in zip(df_sub['lon'], df_sub['lat'])]
        
        # Create GeoDataFrame with WGS84 CRS
        gdf = gpd.GeoDataFrame(
            df_sub[["node_id", "amenity"]],
            geometry=geometry,
            crs="EPSG:4326"
        )
        
        # Save GeoJSON
        out_path = OUTPUT_DIR / f"node_candidates_{amenity}.geojson"
        gdf.to_file(out_path, driver="GeoJSON")
        
        logger.info(f"Created {len(gdf)} candidate nodes -> {out_path.name}")
        return out_path


# ================= STEP 2: UPLOAD TO GEE =================

class GEEUploader:
    """Upload GeoJSON to Google Earth Engine"""
    
    def __init__(self, project_id):
        self.project_id = project_id
        self._initialize_ee()
    
    def _initialize_ee(self):
        """Initialize Earth Engine with service account"""
        if not SERVICE_ACCOUNT_JSON.exists():
            raise FileNotFoundError(
                f"Service account JSON not found: {SERVICE_ACCOUNT_JSON}\n"
                "Download it from IAM -> Service Accounts -> KEYS -> ADD KEY -> JSON"
            )
        
        credentials = service_account.Credentials.from_service_account_file(
            str(SERVICE_ACCOUNT_JSON),
            scopes=['https://www.googleapis.com/auth/earthengine']
        )
        ee.Initialize(credentials=credentials, project=self.project_id)
    
    def upload_geojson(self, geojson_path: Path, amenity: str) -> str:
        """Upload GeoJSON to GEE and return asset ID"""
        logger.info("="*60)
        logger.info(f"STEP 2: Uploading {amenity.upper()} to GEE")
        logger.info("="*60)
        
        asset_id = f'projects/{self.project_id}/assets/{amenity}_nodes'
        
        # Delete existing asset if it exists to prevent overwrite errors
        try:
            ee.data.getAsset(asset_id)
            logger.info(f"Deleting existing asset: {asset_id}")
            ee.data.deleteAsset(asset_id)
            time.sleep(2)  # Brief pause to ensure deletion completes
        except ee.EEException:
            # Asset doesn't exist, which is fine
            pass
        
        # Read GeoJSON
        gdf = gpd.read_file(geojson_path)
        logger.info(f"Loaded {len(gdf)} nodes from {geojson_path.name}")
        
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
        logger.info(f"Starting upload to {asset_id}...")
        task = ee.batch.Export.table.toAsset(
            collection=fc,
            description=f'{amenity}_nodes_upload',
            assetId=asset_id
        )
        task.start()
        
        # Wait for completion
        logger.info("Waiting for upload to complete...")
        wait_count = 0
        while task.status()['state'] in ['READY', 'RUNNING']:
            wait_count += 1
            if wait_count % 6 == 0:  # Log every 30 seconds
                logger.info(f"Still waiting... ({wait_count * 5}s elapsed)")
            time.sleep(5)
        
        status = task.status()
        if status['state'] == 'COMPLETED':
            logger.info("Upload COMPLETED")
            return asset_id
        else:
            raise RuntimeError(f"Upload failed: {status}")


# ================= STEP 3: GEE ANALYSIS =================

class LandUseAnalyzer:
    """Run land-use feasibility analysis on GEE"""
    
    def __init__(self, project_id):
        self.project_id = project_id
        self.MIN_AREA = MIN_AREA
        self.BUFFER_METERS = BUFFER_METERS
        self._initialize_ee()
        self._load_earth_engine_data()
    
    def _initialize_ee(self):
        """Initialize Earth Engine with service account"""
        if not SERVICE_ACCOUNT_JSON.exists():
            raise FileNotFoundError(
                f"Service account JSON not found: {SERVICE_ACCOUNT_JSON}\n"
                "Download it from IAM -> Service Accounts -> KEYS -> ADD KEY -> JSON"
            )
        
        credentials = service_account.Credentials.from_service_account_file(
            str(SERVICE_ACCOUNT_JSON),
            scopes=['https://www.googleapis.com/auth/earthengine']
        )
        ee.Initialize(credentials=credentials, project=self.project_id)
    
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
        logger.info("="*60)
        logger.info(f"STEP 3: Running GEE Analysis for {amenity.upper()}")
        logger.info("="*60)
        
        nodes = ee.FeatureCollection(asset_id)
        node_count = nodes.size().getInfo()
        logger.info(f"Loaded {node_count} nodes from GEE")
        
        logger.info(f"Processing nodes with {self.BUFFER_METERS}m buffer...")
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
        
        logger.info("="*60)
        logger.info("ANALYSIS RESULTS:")
        logger.info(f"  Total nodes: {node_count}")
        logger.info(f"  Feasible nodes: {feasible_count}")
        logger.info(f"  Nodes with placements: {placement_count}")
        logger.info("="*60)
        
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
        logger.info("="*60)
        logger.info(f"STEP 4: Downloading Results for {amenity.upper()}")
        logger.info("="*60)
        
        # Download feasibility as CSV
        feas_output = output_dir / f"landuse_feasibility_{amenity}.csv"
        logger.info("Downloading feasibility data...")
        
        # Get data as list of features
        feas_data = feasibility_fc.getInfo()['features']
        feas_rows = [f['properties'] for f in feas_data]
        feas_df = pd.DataFrame(feas_rows)
        feas_df.to_csv(feas_output, index=False)
        logger.info(f"Saved -> {feas_output.name}")
        
        # Download placements as GeoJSON
        place_output = output_dir / f"landuse_placements_{amenity}.geojson"
        logger.info("Downloading placement polygons...")
        
        # Check if placements collection is empty to avoid geemap error
        placement_count = placements_fc.size().getInfo()
        if placement_count == 0:
            # Create empty GeoJSON structure
            empty_gdf = gpd.GeoDataFrame(
                {'node_id': [], 'amenity': [], 'free_area_m2': [], 
                 'min_area_req': [], 'distance_m': [], 'patch_area_m2': []},
                geometry=[],
                crs="EPSG:4326"
            )
            empty_gdf.to_file(place_output, driver="GeoJSON")
            logger.info(f"Saved -> {place_output.name} (empty - no placements found)")
        else:
            geemap.ee_export_geojson(placements_fc, filename=str(place_output))
            logger.info(f"Saved -> {place_output.name}")
        
        return feas_output, place_output


# ================= STEP 5: FEASIBILITY FILTER =================

class FeasibilityFilter:
    """Filter and merge feasibility results with node attributes"""
    
    def __init__(self, nodes_parquet_path):
        self.nodes_parquet_path = nodes_parquet_path
    
    def filter_and_merge(self, amenity: str, feas_csv: Path, placements_geojson: Path, output_dir: Path):
        """Filter feasible nodes and merge with node attributes"""
        logger.info("="*60)
        logger.info(f"STEP 5: Filtering Feasible Nodes for {amenity.upper()}")
        logger.info("="*60)
        
        # Load data - reset index to make osmid a column
        nodes = pd.read_parquet(self.nodes_parquet_path).reset_index()
        feas = pd.read_csv(feas_csv)
        placements = gpd.read_file(placements_geojson)
        
        logger.info(f"Loaded {len(nodes)} nodes from parquet")
        logger.info(f"Loaded {len(feas)} feasibility records")
        logger.info(f"Loaded {len(placements)} placement polygons")
        
        # Filter feasibility to this amenity
        feas_amen = feas[feas["amenity"] == amenity].copy()
        
        if feas_amen["feasible"].dtype != bool:
            feas_amen["feasible"] = feas_amen["feasible"].astype(bool)
        
        feasible_only = feas_amen[feas_amen["feasible"]]
        logger.info(f"Found {len(feasible_only)} feasible nodes")
        
        # Handle case with no feasible nodes
        if len(feasible_only) == 0:
            logger.warning(f"No feasible nodes found for {amenity.upper()}")
            logger.info("Skipping merge and creating empty output files...")
            
            # Save empty outputs
            out_feas_nodes = output_dir / f"gee_feasible_nodes_{amenity}.csv"
            out_merged = output_dir / f"gee_feasible_nodes_{amenity}_merged.csv"
            out_placements = output_dir / f"gee_placements_{amenity}.geojson"
            
            feasible_only.to_csv(out_feas_nodes, index=False)
            
            # Create empty merged dataframe with expected columns
            empty_merged = pd.DataFrame(columns=['osmid', 'node_id', 'amenity', 'free_area_m2', 'min_area_req', 'feasible'])
            empty_merged.to_csv(out_merged, index=False)
            
            # Placements already saved as empty in download step
            
            logger.info("Saved empty outputs")
            return out_feas_nodes, out_merged, out_placements
        
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
        
        # Filter placements by amenity (handle empty case)
        if len(placements) > 0 and "amenity" in placements.columns:
            placements_clean = placements[placements["amenity"] == amenity]
        else:
            placements_clean = placements
        
        # Save outputs
        out_feas_nodes = output_dir / f"gee_feasible_nodes_{amenity}.csv"
        out_merged = output_dir / f"gee_feasible_nodes_{amenity}_merged.csv"
        out_placements = output_dir / f"gee_placements_{amenity}.geojson"
        
        feasible_only.to_csv(out_feas_nodes, index=False)
        merged.to_csv(out_merged, index=False)
        placements_clean.to_file(out_placements, driver="GeoJSON")
        
        logger.info(f"Saved feasible nodes -> {out_feas_nodes.name}")
        logger.info(f"Saved merged data -> {out_merged.name}")
        logger.info(f"Saved placements -> {out_placements.name}")
        
        # Show sample
        logger.info("Top 5 feasible nodes:")
        cols_show = ['osmid', 'free_area_m2', 'min_area_req', 'feasible']
        if 'walkability' in merged.columns:
            cols_show.append('walkability')
        if 'accessibility_score' in merged.columns:
            cols_show.append('accessibility_score')
        cols_show = [c for c in cols_show if c in merged.columns]
        logger.info("\n" + merged[cols_show].head().to_string(index=False))
        
        return out_feas_nodes, out_merged, out_placements


# ================= MAIN PIPELINE =================

class LandUsePipeline:
    """Complete land-use feasibility pipeline orchestrator"""
    
    def __init__(self):
        self.placement_builder = AmenityPlacementBuilder(BEST_CANDIDATE_PATH, NODES_PARQUET_PATH)
        self.gee_uploader = GEEUploader(PROJECT_ID)
        self.analyzer = LandUseAnalyzer(PROJECT_ID)
        self.downloader = GEEDownloader()
        self.filter = FeasibilityFilter(NODES_PARQUET_PATH)
    
    def process_amenity(self, amenity: str):
        """Run complete pipeline for a single amenity"""
        start_time = time.time()
        
        logger.info("#"*60)
        logger.info(f"# LAND-USE FEASIBILITY PIPELINE: {amenity.upper()}")
        logger.info("#"*60)
        
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
            logger.info("="*60)
            logger.info(f"PIPELINE COMPLETED for {amenity.upper()} in {elapsed:.1f}s")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR processing {amenity}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def process_all_amenities(self):
        """Run pipeline for all amenities"""
        amenities = list(MIN_AREA.keys())
        
        logger.info("#"*60)
        logger.info(f"# LAND-USE BATCH PIPELINE: {len(amenities)} AMENITIES")
        logger.info("#"*60)
        
        results = {}
        for amenity in amenities:
            success = self.process_amenity(amenity)
            results[amenity] = success
        
        # Summary
        logger.info("="*60)
        logger.info("BATCH PIPELINE SUMMARY")
        logger.info("="*60)
        for amenity, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"  {amenity:15} | {status}")
        
        return results


# ================= INTEGRATION FOR HYBRID GA =================

class LanduseFeasibilityIntegration:
    """
    Integration class for filtering candidate placements from hybrid_ga.py
    based on landuse feasibility using Google Earth Engine.
    
    Usage from hybrid_ga.py:
        from landuse_pipeline.run_feasibility import LanduseFeasibilityIntegration
        
        integrator = LanduseFeasibilityIntegration(nodes_parquet_path)
        filtered_placements = integrator.filter_candidate_placements(placements_dict)
    """
    
    def __init__(self, nodes_parquet_path: Path = None):
        """Initialize the integration with paths to required data."""
        self.nodes_parquet_path = nodes_parquet_path or NODES_PARQUET_PATH
        self.project_id = PROJECT_ID
        self.min_area = MIN_AREA
        self.buffer_meters = BUFFER_METERS
        self._ee_initialized = False
        self._nodes_df = None
        
    def _ensure_ee_initialized(self):
        """Lazy initialization of Earth Engine."""
        if self._ee_initialized:
            return
            
        if not SERVICE_ACCOUNT_JSON.exists():
            raise FileNotFoundError(
                f"Service account JSON not found: {SERVICE_ACCOUNT_JSON}\n"
                "Download it from IAM -> Service Accounts -> KEYS -> ADD KEY -> JSON"
            )
        
        credentials = service_account.Credentials.from_service_account_file(
            str(SERVICE_ACCOUNT_JSON),
            scopes=['https://www.googleapis.com/auth/earthengine']
        )
        ee.Initialize(credentials=credentials, project=self.project_id)
        self._ee_initialized = True
        
        # Load ESA WorldCover data
        self.worldCover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
        self.freeMask = (self.worldCover.eq(30)
                         .Or(self.worldCover.eq(40))
                         .Or(self.worldCover.eq(60)))
        self.pixelArea = ee.Image.pixelArea()
        
    def _load_nodes_df(self):
        """Load nodes dataframe lazily."""
        if self._nodes_df is None:
            self._nodes_df = pd.read_parquet(self.nodes_parquet_path).reset_index()
        return self._nodes_df
    
    def _process_node_feasibility(self, feature):
        """Process a single node to determine feasibility."""
        f = ee.Feature(feature)
        amenity = ee.String(f.get('amenity'))
        
        bufferGeom = f.geometry().buffer(self.buffer_meters, 1)
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
        minReq = ee.Number(ee.Dictionary(self.min_area).get(amenity))
        feasible = freeAreaM2.gte(minReq)
        
        return f.set({
            'free_area_m2': freeAreaM2,
            'min_area_req': minReq,
            'feasible': feasible
        })
    
    def filter_candidate_placements(self, placements: dict) -> dict:
        """
        Filter candidate placements based on landuse feasibility.
        
        Args:
            placements: Dict mapping amenity type to tuple/list of node IDs
                       Example: {'hospital': ('123', '456'), 'school': ('789',)}
        
        Returns:
            Dict with same structure but only containing feasible node IDs
        """
        logger.info("=" * 60)
        logger.info("LANDUSE FEASIBILITY INTEGRATION")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Initialize GEE
        self._ensure_ee_initialized()
        
        # Load nodes data
        nodes_df = self._load_nodes_df()
        
        filtered_placements = {}
        
        for amenity, node_ids in placements.items():
            if not node_ids:
                filtered_placements[amenity] = tuple()
                continue
                
            # Check if amenity has min area requirement
            if amenity not in self.min_area:
                logger.warning(f"No min area requirement for {amenity}, keeping all nodes")
                filtered_placements[amenity] = tuple(node_ids)
                continue
            
            logger.info(f"Processing {amenity}: {len(node_ids)} candidate nodes")
            
            try:
                feasible_ids = self._check_amenity_feasibility(amenity, node_ids, nodes_df)
                filtered_placements[amenity] = tuple(sorted(feasible_ids))
                
                removed_count = len(node_ids) - len(feasible_ids)
                if removed_count > 0:
                    logger.info(f"  {amenity}: Removed {removed_count} infeasible nodes, kept {len(feasible_ids)}")
                else:
                    logger.info(f"  {amenity}: All {len(feasible_ids)} nodes are feasible")
                    
            except Exception as e:
                logger.error(f"Error checking feasibility for {amenity}: {e}")
                # On error, keep all nodes rather than losing data
                filtered_placements[amenity] = tuple(node_ids)
        
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"LANDUSE FILTERING COMPLETE in {elapsed:.1f}s")
        logger.info("=" * 60)
        
        return filtered_placements
    
    def _check_amenity_feasibility(self, amenity: str, node_ids: tuple, nodes_df: pd.DataFrame) -> list:
        """Check feasibility for a single amenity type and return feasible node IDs."""
        # Convert node IDs to integers for matching
        node_ids_int = [int(nid) for nid in node_ids]
        
        # Get node coordinates
        df_sub = nodes_df[nodes_df['osmid'].isin(node_ids_int)].copy()
        
        if df_sub.empty:
            logger.warning(f"No nodes found in parquet for {amenity}")
            return []
        
        # Build GEE FeatureCollection directly (no file I/O)
        features = []
        for _, row in df_sub.iterrows():
            point = ee.Geometry.Point([float(row['lon']), float(row['lat'])])
            feature = ee.Feature(point, {
                'node_id': int(row['osmid']),
                'amenity': amenity
            })
            features.append(feature)
        
        fc = ee.FeatureCollection(features)
        
        # Run feasibility analysis
        results = fc.map(self._process_node_feasibility)
        
        # Get feasible node IDs
        feasible_fc = results.filter(ee.Filter.eq('feasible', 1))
        feasible_data = feasible_fc.getInfo()
        
        feasible_ids = []
        for f in feasible_data.get('features', []):
            node_id = f['properties'].get('node_id')
            if node_id is not None:
                feasible_ids.append(str(node_id))
        
        return feasible_ids


# ================= ENTRY POINT =================

def main():
    if len(sys.argv) < 2:
        logger.info("Usage:")
        logger.info("  python run_feasibility.py <amenity>     # Process single amenity")
        logger.info("  python run_feasibility.py --all         # Process all amenities")
        logger.info(f"\nAvailable amenities: {list(MIN_AREA.keys())}")
        sys.exit(1)
    
    pipeline = LandUsePipeline()
    
    if sys.argv[1] == '--all':
        pipeline.process_all_amenities()
    else:
        amenity = sys.argv[1]
        if amenity not in MIN_AREA:
            logger.error(f"Unknown amenity '{amenity}'")
            logger.info(f"Available: {list(MIN_AREA.keys())}")
            sys.exit(1)
        pipeline.process_amenity(amenity)


if __name__ == "__main__":
    main()