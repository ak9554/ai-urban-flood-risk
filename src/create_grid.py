import geopandas as gpd
from shapely.geometry import box
from pathlib import Path


BOUNDARY_FILE = Path("data/processed/ncr_7_districts.geojson")
OUTPUT_FILE = Path("data/processed/ncr_1km_grid.geojson")

METRIC_CRS = "EPSG:32643"
GRID_SIZE = 1000


# ---------------------------------------------------------
# 1. Load study-area boundary
# ---------------------------------------------------------

boundary = gpd.read_file(BOUNDARY_FILE)

print("Study-area districts:", len(boundary))
print("Original CRS:", boundary.crs)


# ---------------------------------------------------------
# 2. Convert boundary to metric CRS
# ---------------------------------------------------------

boundary_metric = boundary.to_crs(METRIC_CRS)

print("Metric CRS:", boundary_metric.crs)


# Merge all seven districts
study_area = boundary_metric.geometry.union_all()


# ---------------------------------------------------------
# 3. Find bounding box
# ---------------------------------------------------------

min_x, min_y, max_x, max_y = study_area.bounds

print()
print("Study-area bounds (metres):")
print("Min X:", min_x)
print("Min Y:", min_y)
print("Max X:", max_x)
print("Max Y:", max_y)


# ---------------------------------------------------------
# 4. Create 1 km × 1 km grid
# ---------------------------------------------------------

grid_cells = []

x = min_x

while x < max_x:

    y = min_y

    while y < max_y:

        cell = box(
            x,
            y,
            x + GRID_SIZE,
            y + GRID_SIZE
        )

        if cell.intersects(study_area):
            grid_cells.append(cell)

        y += GRID_SIZE

    x += GRID_SIZE


print()
print("Generated cells:", len(grid_cells))


# ---------------------------------------------------------
# 5. Create GeoDataFrame
# ---------------------------------------------------------

grid = gpd.GeoDataFrame(
    {"geometry": grid_cells},
    crs=METRIC_CRS
)


# ---------------------------------------------------------
# 6. Clip cells to study area
# ---------------------------------------------------------

grid["geometry"] = grid.geometry.intersection(
    study_area
)

grid = grid[
    ~grid.geometry.is_empty
].copy()


# ---------------------------------------------------------
# 7. Create grid IDs
# ---------------------------------------------------------

grid = grid.reset_index(drop=True)

grid["grid_id"] = [
    f"NCR_{i:04d}"
    for i in range(1, len(grid) + 1)
]


# ---------------------------------------------------------
# 8. Calculate centroid WHILE STILL IN METRIC CRS
# ---------------------------------------------------------

centroids = grid.geometry.centroid

grid["centroid_x"] = centroids.x
grid["centroid_y"] = centroids.y


# ---------------------------------------------------------
# 9. Convert grid to latitude/longitude
# ---------------------------------------------------------

grid = grid.to_crs("EPSG:4326")


# Convert the metric centroid coordinates
# to latitude/longitude properly.
centroid_gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(
        grid["centroid_x"],
        grid["centroid_y"]
    ),
    crs=METRIC_CRS
)

centroid_gdf = centroid_gdf.to_crs("EPSG:4326")

grid["longitude"] = centroid_gdf.geometry.x
grid["latitude"] = centroid_gdf.geometry.y


# Remove temporary metric centroid columns
grid = grid.drop(
    columns=["centroid_x", "centroid_y"]
)


# ---------------------------------------------------------
# 10. Save
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

grid.to_file(
    OUTPUT_FILE,
    driver="GeoJSON"
)


print()
print("Grid creation complete.")
print("Number of grid cells:", len(grid))
print("Saved:", OUTPUT_FILE)

print()
print("First five grid cells:")

print(
    grid[
        [
            "grid_id",
            "latitude",
            "longitude"
        ]
    ].head()
)