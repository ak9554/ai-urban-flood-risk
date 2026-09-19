import geopandas as gpd

path = "data/raw/boundaries/DISTRICT_BOUNDARY.shp"

gdf = gpd.read_file(path)

print("Number of districts:", len(gdf))
print("\nColumns:")
print(gdf.columns.tolist())

print("\nCoordinate system:")
print(gdf.crs)

print("\nFirst 5 districts:")
print(gdf[["STATE_UT", "DISTRICT", "DIST_LGD"]].head())