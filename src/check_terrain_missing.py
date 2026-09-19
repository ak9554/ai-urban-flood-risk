import geopandas as gpd
import pandas as pd
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

GRID_FILE = Path("data/processed/ncr_1km_grid.geojson")
TERRAIN_FILE = Path("data/processed/ncr_terrain_features.csv")
BOUNDARY_FILE = Path("data/processed/ncr_7_districts.geojson")


# ============================================================
# LOAD DATA
# ============================================================

grid = gpd.read_file(GRID_FILE)
terrain = pd.read_csv(TERRAIN_FILE)
boundary = gpd.read_file(BOUNDARY_FILE)


# ============================================================
# FIND GRID CELLS WITH MISSING TERRAIN
# ============================================================

missing_ids = terrain.loc[
    terrain["elevation_mean"].isna(),
    "grid_id"
]

missing_grid = grid[
    grid["grid_id"].isin(missing_ids)
].copy()


# ============================================================
# BASIC SUMMARY
# ============================================================

print("=" * 60)
print("TERRAIN MISSING-VALUE INVESTIGATION")
print("=" * 60)

print("Total grid cells:", len(grid))

print(
    "Cells with terrain:",
    terrain["elevation_mean"].notna().sum()
)

print(
    "Cells missing terrain:",
    terrain["elevation_mean"].isna().sum()
)

print()

missing_percentage = (
    terrain["elevation_mean"].isna().mean() * 100
)

print(
    "Missing percentage:",
    round(missing_percentage, 2),
    "%"
)


# ============================================================
# ASSIGN EACH MISSING CELL TO ONE DISTRICT
# USING THE GRID CELL CENTROID
# ============================================================

# Make sure both layers use the same CRS
boundary = boundary.to_crs(grid.crs)

# Calculate centroid of each missing grid cell
missing_grid["centroid"] = missing_grid.geometry.centroid

# Create a GeoDataFrame containing only the centroid points
missing_centroids = gpd.GeoDataFrame(
    missing_grid[["grid_id"]],
    geometry=missing_grid["centroid"],
    crs=grid.crs
)


# ============================================================
# SPATIAL JOIN
# ============================================================

missing_with_district = gpd.sjoin(
    missing_centroids,
    boundary[["DISTRICT", "geometry"]],
    how="left",
    predicate="within"
)


# ============================================================
# CHECK DISTRICT COUNTS
# ============================================================

print()
print("Missing cells by district:")

district_counts = (
    missing_with_district["DISTRICT"]
    .value_counts(dropna=False)
)

print(district_counts)

print()

assigned_count = missing_with_district["DISTRICT"].notna().sum()
unassigned_count = missing_with_district["DISTRICT"].isna().sum()

print("Missing cells assigned to a district:", assigned_count)
print("Missing cells without district:", unassigned_count)

print(
    "Total after district assignment:",
    district_counts.sum()
)


# ============================================================
# GEOGRAPHIC RANGE
# ============================================================

print()
print("Missing-cell latitude range:")

print(
    "Minimum:",
    missing_grid["latitude"].min()
)

print(
    "Maximum:",
    missing_grid["latitude"].max()
)

print()
print("Missing-cell longitude range:")

print(
    "Minimum:",
    missing_grid["longitude"].min()
)

print(
    "Maximum:",
    missing_grid["longitude"].max()
)


# ============================================================
# SAMPLE MISSING CELLS
# ============================================================

print()
print("First 20 missing grid cells:")

print(
    missing_grid[
        [
            "grid_id",
            "latitude",
            "longitude"
        ]
    ].head(20)
)


# ============================================================
# FINAL CHECK
# ============================================================

print()
print("=" * 60)
print("CONSISTENCY CHECK")
print("=" * 60)

total_missing = terrain["elevation_mean"].isna().sum()
total_district_assigned = missing_with_district["DISTRICT"].notna().sum()

print("Terrain missing cells:", total_missing)
print("District-assigned cells:", total_district_assigned)

if total_missing == total_district_assigned:
    print("SUCCESS: Every missing cell was assigned to exactly one district.")
else:
    print("WARNING: Some missing cells could not be assigned to a district.")

print()
print("Investigation complete.")