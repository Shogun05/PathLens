# PathLens Land-Use Feasibility Workflow

**Directory:** `GEE-landscape/`  
**Purpose:** Take GA/MILP output (`best_candidate.json` + `optimized_nodes_with_scores.csv`), check land-use feasibility in Google Earth Engine (GEE), and produce filtered nodes + placement polygons per amenity (e.g., hospital, school).

Required files in `GEE-landscape/`:

- `best_candidate.json` – GA/MILP result with amenity → osmid mapping.
- `optimized_nodes_with_scores.csv` – node table with `osmid, lon, lat, ...`.
- `amenity_placement.py` – builds candidate GeoJSON for a chosen amenity.
- `geojsontocsv.py` – converts candidate GeoJSON → CSV for GEE upload.
- `feasibilityfilter.py` – post‑GEE integration script.

You also need a GEE project (e.g., `your_project_name`) and ESA WorldCover v200 available in GEE.  
ESA WorldCover provides global 10 m land‑cover classes, including grassland, cropland, and bare/sparse used as “free land.” [web:21]

---

## 1. From optimizer output to amenity GeoJSON

### 1.1. Ensure inputs are present

In `GEE-landscape/`:

- `best_candidate.json` contains strings like:

```

{
"candidate": "bus_station:10020025539,10991934818|hospital:10035832527,10216399582,...",
...
}

```

- `optimized_nodes_with_scores.csv` has at least:

```

osmid,y,x,street_count,lon,lat,highway,...

```

Here `osmid` is the node ID; `lon` and `lat` are coordinates.

### 1.2. Generate candidates for one amenity

From a terminal inside `GEE-landscape/`:

```


# Example: hospital run

python amenity_placement.py hospital

```

`amenity_placement.py` should:

1. Parse `best_candidate.json`.
2. Extract osmids for the given amenity (e.g., `"hospital"`).
3. Look up `lon, lat` in `optimized_nodes_with_scores.csv`.
4. Write `node_candidates_<amenity>.geojson`, e.g.:

```

node_candidates_hospital.geojson

```

Each feature in this GeoJSON has:

- `properties`: `node_id` (int), `amenity` (string).
- `geometry`: `Point(lon, lat)`.

This is the spatial input for GEE’s ESA WorldCover analysis. [web:21]

---

## 2. GeoJSON → CSV (for GEE table upload)

GEE’s Code Editor uploads CSV and shapefiles as tables; converting your GeoJSON to CSV makes upload straightforward. [web:20]

From the terminal:

```

python geojsontocsv.py node_candidates_hospital.geojson

```

`geojsontocsv.py` should:

1. Read the GeoJSON.
2. Extract `node_id`, `amenity`, `lon`, `lat` from the geometry.
3. Write:

```

node_candidates_hospital.csv

```

with columns:

```

node_id, amenity, lon, lat

```

---

## 3. Manual GEE workflow (per amenity)

### 3.1. Upload CSV as a table asset

1. Open the GEE Code Editor in a browser.
2. Go to **Assets → NEW → Table upload → CSV file (.csv)**. [web:20]
3. Select `node_candidates_hospital.csv` from `GEE-landscape/`.
4. Set asset ID, for example:

```

projects/your_project_name/assets/hospital_nodes

```

5. In the geometry settings, choose:
- X column: `lon`
- Y column: `lat`
6. Click **Upload** and wait until the upload finishes (check the *Tasks* tab).

Now you have a `FeatureCollection` of candidate nodes for this amenity.

### 3.2. Run the feasibility script

Create a new script in the Code Editor, paste your **amenity feasibility code**, and set:

```

var AMENITY_ASSET = 'projects/your_project_name/assets/hospital_nodes';

```

Key elements of the script:

- Load nodes:

```

var nodes = ee.FeatureCollection(AMENITY_ASSET);

```

- Load ESA WorldCover and build free‑land mask (grassland 30, cropland 40, bare/sparse 60). [web:21]

```

var worldCover = ee.ImageCollection('ESA/WorldCover/v200')
.first()
.select('Map');

var freeMask = worldCover.eq(30)
.or(worldCover.eq(40))
.or(worldCover.eq(60));

```

- Compute free land within a buffer (e.g., 200 m) for each node via `pixelArea().reduceRegion(...)`. [web:18]
- Compare total free area with a `MIN_AREA` threshold for the amenity; set `feasible`, `free_area_m2`, etc.
- For feasible nodes, call `reduceToVectors` on the masked free‑land image to get contiguous free‑land polygons. [web:24][web:68]
- Choose the nearest polygon as `best_patch` and attach `distance_m` and `patch_area_m2`.

The script exports two tables to Google Drive:

```

Export.table.toDrive({
collection: feasibility,      // node_id, amenity, free_area_m2, feasible, ...
description: 'pathlens_feasibility_single_amenity',
fileFormat: 'CSV'
});

Export.table.toDrive({
collection: placements,       // polygons with node_id, distance_m, patch_area_m2
description: 'pathlens_placements_single_amenity',
fileFormat: 'GeoJSON'
});

```

In the **Tasks** tab:

1. Click **RUN** on both exports.
2. Wait for completion.
3. Download:
   - `pathlens_feasibility_single_amenity.csv`
   - `pathlens_placements_single_amenity.geojson`

Place these files back into `GEE-landscape/`.

---

## 4. Post‑GEE integration (filtering and merging)

Now run the post‑processing script to integrate GEE results with your PathLens node table.

From the terminal:

```

python feasibilityfilter.py

```

`feasibilityfilter.py` assumes:

- `optimized_nodes_with_scores.csv` is in `GEE-landscape/`.
- `pathlens_feasibility_single_amenity.csv` and `pathlens_placements_single_amenity.geojson` are present (downloaded from Drive).
- An `AMENITY` variable inside the script is set (e.g., `"hospital"`).

The script:

1. Loads:
   - `optimized_nodes_with_scores.csv`
   - `pathlens_feasibility_single_amenity.csv`
   - `pathlens_placements_single_amenity.geojson`
2. Filters feasibility to `amenity == AMENITY` and `feasible == True` (or 1).
3. Joins feasible nodes with the original node table on `osmid == node_id`.
4. Cleans placement polygons for this amenity.

It writes three key outputs:

- `gee_feasible_nodes_<amenity>.csv`  
  List of `node_id` plus GEE attributes (`free_area_m2`, `min_area_req`, etc.) for feasible nodes.

- `gee_feasible_nodes_<amenity>_merged.csv`  
  Same as above, but merged with `optimized_nodes_with_scores.csv` so you have `lon`, `lat`, `walkability`, `accessibility_score`, etc., along with GEE feasibility metrics.

- `gee_placements_<amenity>.geojson`  
  Polygons of actual free‑land patches with properties:

```

node_id, amenity, free_area_m2, min_area_req, distance_m, patch_area_m2, geometry

```

These outputs become:

- Constraints for your **second‑pass GA/MILP** (use only feasible nodes per amenity as decision variables).  
- Geometries for **maps/visualization** and downstream planning.

---

## 5. Repeating for other amenities

For another amenity (e.g., `school`):

1. Run:

```

python amenity_placement.py school
python geojsontocsv.py node_candidates_school.geojson

```

2. Upload `node_candidates_school.csv` to GEE as `projects/.../school_nodes`.
3. In the GEE script, change:

```

var AMENITY_ASSET = 'projects/.../school_nodes';

```

and adjust `MIN_AREA['school']` and buffer radius if needed.
4. Export the new feasibility + placements, download them, place in `GEE-landscape/`.
5. Set `AMENITY = "school"` inside `feasibilityfilter.py` and run:

```

python feasibilityfilter.py

```

You will get the school‑specific feasible node list and placement polygons without affecting the hospital outputs.

---

## 6. Using results in PathLens optimizer

Within your GA/MILP code (outside this folder), you can now:

- Read `gee_feasible_nodes_<amenity>.csv` to get the feasible `node_id` list per amenity.
- Restrict candidate sets:

```

feasible_hospitals = set(pd.read_csv("gee_feasible_nodes_hospital.csv")["node_id"])
hospital_candidates = [n for n in original_hospital_nodes if n in feasible_hospitals]

```

- Optionally, use `gee_feasible_nodes_<amenity>_merged.csv` to incorporate `free_area_m2`, `distance_m`, or `patch_area_m2` into your scoring functions.

This keeps:

- **Decision variables:** node IDs, same as your existing optimizer.  
- **Constraints:** feasibility from land‑use (via ESA WorldCover) encoded as a pre‑filter on candidate nodes. [web:21]  
- **Artifacts:** polygons from GEE used only for validation, visualization, and execution, not as decision variables.

