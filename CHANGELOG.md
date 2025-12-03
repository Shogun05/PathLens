# PathLens Changelog

## 2025-12-03 - Major Bug Fixes and Project Restructuring

### Critical Bug Fixes

#### 1. Edge Length Calculation Bug (71,000x multiplier)
**Problem**: Network edges were showing distances of ~5 million meters instead of ~63 meters average.

**Root Cause**: 
- `ox.distance.add_edge_lengths()` was calculating lengths from edge geometries stored in EPSG:4326 (lat/lon degrees) instead of EPSG:32643 (UTM meters)
- For edges without geometry, it used lat/lon coordinates directly, treating degrees as meters
- Result: 1 degree ≈ 111km was treated as 1 meter

**Solution**: 
- Skipped `add_edge_lengths_compat()` call since raw graph already has correct edge lengths
- Added length preservation logic in `nodes_edges_to_gdfs()` to maintain metric values during CRS reprojection
- Final edge statistics: Mean=63.55m, Median=44.19m, Max=1,603m (correct!)

**Files Modified**:
- `pipeline/graph_build.py`: Lines 115-128 (commented out problematic edge length calculation)
- `pipeline/graph_build.py`: Lines 137-149 (added length_m preservation during reprojection)

#### 2. Coverage Calculation Showing 0% for All Amenities
**Problem**: All amenity coverage showing 0.0 despite POIs being present in the data.

**Root Cause**: Amenity search only checked the 'amenity' column, missing:
- Supermarkets in 'shop' column (3,938 POIs)
- Parks in 'leisure' column (9,524 POIs)

**Solution**: 
- Modified amenity search to check across multiple columns: amenity, shop, and leisure
- Lines 197-207 in `scoring.py`: Added mask checking all three columns

**Results After Fix**:
- School: 50.5% coverage (10,041/19,876 nodes)
- Hospital: 77.1% coverage (15,325/19,876 nodes)
- Park: 69.7% coverage (13,859/19,876 nodes)
- Pharmacy: 19.8% coverage (3,945/19,876 nodes)
- Supermarket: 30.7% coverage (6,102/19,876 nodes)
- Bus Station: 5.0% coverage (996/19,876 nodes)

#### 3. Cache Boolean Check Error
**Problem**: `ValueError: The truth value of a GeoDataFrame is ambiguous` when loading cached data.

**Root Cause**: Used walrus operator in boolean context with GeoDataFrame return values.

**Solution**: 
- Changed from `if not args.force and (cached_nodes := load_from_cache(...)):`
- To: `cached_nodes = load_from_cache(...); if not args.force and cached_nodes is not None:`
- Applied to both structure and distance cache checks

**Files Modified**:
- `pipeline/scoring.py`: Lines 389-390, 411-412

#### 4. Distance Cutoff Too Restrictive
**Problem**: 1200m distance cutoff resulted in 99.5% of nodes having infinite distances (only 906/182,185 nodes reachable).

**Solution**: Changed `amenity_distance_cutoff_m: 1200` → `amenity_distance_cutoff_m: null` in config.yaml

### Performance Improvements

#### Caching System Implementation
Added pickle-based caching for expensive computations:

**Structure Cache**: 
- Caches betweenness centrality (previously ~10 minutes per run)
- Cache key: MD5 hash of graph path
- File: `data/analysis/.cache/structure_<hash>.pkl`

**Distance Cache**:
- Caches network distance computations (previously ~15 minutes per run)
- Cache key: MD5 hash of (graph path, POI mapping, amenity types, cutoff)
- File: `data/analysis/.cache/distances_<hash>.pkl`

**Files Modified**:
- `pipeline/scoring.py`: Lines 25-48 (cache functions)
- `pipeline/scoring.py`: Lines 380-403 (cache implementation)

### Project Restructuring

#### Directory Organization
**Before**:
```
/home/shogun/Documents/5thsem/
├── convert_amenities_to_geojson.py
├── data_collection.py
├── fetch_bengaluru_amenities.py
├── graph_build.py (ROOT - had fixes)
├── manage_cache.py
├── scoring.py (ROOT - had fixes)
├── visualize.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── analysis/
└── scripts/
    ├── graph_build.py (outdated)
    ├── scoring.py (outdated)
    ├── cache/
    └── data/ (redundant, old data)
```

**After**:
```
/home/shogun/Documents/5thsem/
├── run_pipeline.py
├── config.yaml
├── pipeline/
│   ├── convert_amenities_to_geojson.py
│   ├── data_collection.py
│   ├── fetch_bengaluru_amenities.py
│   ├── graph_build.py (CANONICAL)
│   ├── manage_cache.py
│   ├── scoring.py (CANONICAL)
│   └── visualize.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── analysis/
└── scripts/
    └── cache/
```

**Changes Made**:
1. Moved all 7 Python processing scripts to `pipeline/` directory
2. Deleted redundant `scripts/data/` (contained old, incorrect data from before bug fixes)
3. Deleted duplicate `scripts/graph_build.py` and `scripts/scoring.py`
4. Updated `run_pipeline.py` to reference `pipeline/` directory
5. Updated path references in all pipeline scripts to correctly navigate from `pipeline/` to project root

**Files Modified for Path Updates**:
- `run_pipeline.py`: Added `pipeline_dir` variable, updated all script paths
- `pipeline/fetch_bengaluru_amenities.py`: Fixed cache directory paths
- `pipeline/convert_amenities_to_geojson.py`: Fixed input/output paths
- `pipeline/manage_cache.py`: Fixed project directory navigation

### Data Quality Improvements

#### Graph Statistics (After Fixes)
- Nodes: 19,876 (simplified from 19,876 original)
- Edges: 59,914
- Edge lengths: Mean=63.55m, Median=44.19m, Min=0.59m, Max=1,603m
- CRS: EPSG:32643 (UTM Zone 43N for Bengaluru)

#### POI Statistics
- Total POIs: 243,905
  - amenity=*: 144,222 (59%)
  - shop=*: 71,410 (29%)
  - leisure=*: 29,211 (12%)
  - transit: 453 (0.19%)

#### Walkability Scores (Final Results)
- Structure Score: Mean=0.13, Median=0.13
- Accessibility Score: Mean=0.04, Median=0.01
- Average Travel Time: 4.9 minutes to nearest amenities

### Technical Details

#### Coordinate Reference Systems
- Input data: EPSG:4326 (WGS 84, geographic coordinates)
- Processing: EPSG:32643 (WGS 84 / UTM zone 43N, projected meters)
- Storage: EPSG:4326 (for visualization compatibility)
- **Key Fix**: Preserve metric lengths when reprojecting back to EPSG:4326

#### Distance Calculation
- Algorithm: NetworkX `multi_source_dijkstra_path_length`
- Weight: Edge length in meters
- Cutoff: None (changed from 1200m)
- Source nodes per amenity: 400-15,000 depending on amenity type

#### Amenity Thresholds (for Coverage Calculation)
- School: 800m
- Hospital: 1,000m
- Pharmacy: 600m
- Supermarket: 800m
- Bus Station: 600m
- Park: 700m

### Breaking Changes
None - all changes are bug fixes and internal reorganization.

### Migration Guide
If you have local modifications:
1. Move any custom scripts to `pipeline/` directory
2. Update script paths in `run_pipeline.py` to use `pipeline_dir`
3. Clear old caches: `rm -rf data/analysis/.cache/*`
4. Rebuild graph: Re-run `run_pipeline.py` with `--force` flag

### Files to Delete
The following files are now redundant and can be safely deleted:
- `scripts/graph_build.py` (outdated version)
- `scripts/scoring.py` (outdated version)
- `scripts/data/` (entire directory - contained old, incorrect data)

### Known Issues
None currently. All critical bugs have been resolved.

### Next Steps
- Consider adding visualization for edge length distribution
- Add validation checks for CRS consistency
- Document the correct workflow for adding new amenity types
- Add unit tests for distance calculations
