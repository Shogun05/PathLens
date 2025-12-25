import geopandas as gpd

gdf = gpd.read_file("node_candidates_school.geojson")

# Ensure these columns exist; adapt names if needed
gdf["lon"] = gdf.geometry.x
gdf["lat"] = gdf.geometry.y

gdf[["node_id", "amenity", "lon", "lat"]].to_csv(
    "node_candidates_school.csv",
    index=False
)
print("Wrote node_candidates_school.csv")