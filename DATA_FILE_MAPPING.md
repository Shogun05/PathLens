Directory
|
-data-pipeline/
|
-compute_scores.py
--Cache of computed metrics : cache/*.pkl
--Pipeline configuration : configs/base.yaml
--POI category mapping : data/analysis/poi_mapping.parquet
--Source POIs data : data/raw/pois.geojson or data/raw/pois.parquet
-visualize.py
--Source POIs data : data/raw/pois.geojson
--Processed nodes data : data/analysis/nodes.parquet
--POI mapping : data/analysis/poi_mapping.parquet
-collect_osm_data.py
--Pipeline configuration : configs/base.yaml
-convert_amenities.py
--Raw OSM JSON data : data/raw/*.json
-build_graph.py
--Base POIs data : data/raw/pois.parquet
--Optimized POIs data : data/analysis/optimized_pois.parquet
--Additional network layers : data/network/*.parquet
|
-optimization-pipeline/
|
-run_optimization.py
--Optimization configuration : configs/base.yaml
-train_pnmlr.py
--Training nodes data : data/analysis/baseline_nodes_with_scores.parquet
-run_optimized_scoring.py
--Baseline POIs : data/raw/pois.parquet
--Optimized POIs : data/analysis/optimized_pois.geojson
-hybrid_ga.py
--Genetic Algorithm configuration : configs/base.yaml
--High travel demand nodes : data/analysis/high_travel_nodes.csv
--Network nodes : data/analysis/nodes.parquet
--Solution checkpoints : data/analysis/solutions/*.json
-pnmlr_model.py
--Model state dictionary : models/pnmlr_model.pkl
-list_optimizable_nodes.py
--Input network nodes : data/analysis/nodes.parquet
-hybrid_milp_refiner.py
--Refinement cache : cache/milp_refinement.json
-generate_solution_map.py
--Best candidate solution : data/analysis/best_candidate.json
--Network nodes : data/analysis/nodes.parquet
--POI mapping : data/analysis/poi_mapping.parquet
-visualize_amenity_paths.py
--Best candidate solution : data/analysis/best_candidate.json
-milp_placement.py
--MILP configuration : configs/base.yaml
--Network nodes : data/analysis/nodes.parquet
--Candidate placements : data/analysis/candidates.csv
|
-landuse-pipeline/
|
-utils.py
--Node candidates for schools : node_candidates_school.geojson
-amenity_placement.py
--Placement data : data/analysis/placements.json
--Network nodes : data/analysis/nodes.csv
-feasibility_filter.py
--Network nodes : data/analysis/nodes.csv
--Feasibility metrics : data/analysis/feasibility.csv
--Proposed placements : data/analysis/placements.geojson
-run_feasibility.py
--Best candidate solution : data/analysis/best_candidate.json
--Network nodes : data/analysis/nodes.parquet
--Site feasibility data : data/analysis/feasibility.csv
|
-GraphBuilder/
|
-downloader/main.py
--User preferences : GraphBuilder/downloader/prefs.json
-model/dataloader.py
--Graph neighbors data : GraphBuilder/data/region_*_refine_gt_graph.p
--Graph sample points : GraphBuilder/data/region_*_refine_gt_graph_samplepoints.json
-sat2graph_converter.py
--Satellite graph structure : models/satellite_graph.p
--Graph metadata : models/satellite_metadata.json
|
-backend/
|
-main.py
--Network nodes (Parquet) : data/analysis/nodes.parquet
--Network nodes (CSV) : data/analysis/nodes.csv
--Metrics summary : data/analysis/metrics_summary.json
--Baseline/Optimized GeoJSON : data/analysis/*.geojson
--Best candidate solution : data/analysis/best_candidate.json
--Combined POIs : data/analysis/combined_pois.json
--Job progress : data/analysis/progress.json
|
-Root Directory/
|
-fix_metrics_network.py
--Baseline metrics : data/analysis/baseline_metrics.json
--Optimized metrics : data/analysis/optimized_metrics.json
-test_new_formulas.py
--Configuration : configs/base.yaml
--Scored nodes : data/analysis/baseline_nodes_with_scores.parquet
-clean_candidate.py
--Candidate solution : data/analysis/candidate.json
--Network nodes : data/analysis/nodes.parquet
-download_city.py
--City metadata : data/cities/*/metadata.json
