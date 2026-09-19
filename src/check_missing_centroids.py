import geopandas as gpd
import pandas as pd
from pathlib import Path


GRID_FILE = Path(
    "data/processed/ncr_1km_grid.geojson"
)

TERRAIN_FILE = Path(
    "data/processed/ncr_terrain_features.csv"
)

BOUNDARY_FILE = Path(
    "data/processed/ncr_7_districts.geojson"
)


# ============================================================
# LOAD DATA
# ============================================================

grid = gpd.read_file(GRID_FILE)

terrain = pd.read_csv(TERRAIN_FILE)

boundary = gpd.read_file(BOUNDARY_FILE)


# ============================================================
# FIND MISSING TERRAIN CELLS
# ============================================================

missing_ids = terrain.loc[
    terrain["elevation_mean"].isna(),
    "grid_id"
]

missing_grid = grid[
    grid["grid_id"].isin(missing_ids)
].copy()


# ============================================================
# MAKE SURE CRS MATCHES
# ============================================================

boundary = boundary.to_crs(grid.crs)


# ============================================================
# CREATE CENTROIDS
# ============================================================

missing_grid["centroid"] = (
    missing_grid.geometry.centroid
)

centroids = gpd.GeoDataFrame(
    missing_grid[["grid_id"]],
    geometry=missing_grid["centroid"],
    crs=grid.crs
)


# ============================================================
# CHECK WHETHER CENTROID IS INSIDE NCR
# ============================================================

centroid_join = gpd.sjoin(
    centroids,
    boundary[["DISTRICT", "geometry"]],
    how="left",
    predicate="within"
)


# ============================================================
# CHECK WHETHER GRID CELL ITSELF INTERSECTS NCR
# ============================================================

cell_join = gpd.sjoin(
    missing_grid.drop(
        columns=["centroid"]
    ),
    boundary[["DISTRICT", "geometry"]],
    how="left",
    predicate="intersects"
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 60)
print("MISSING TERRAIN / CENTROID INVESTIGATION")
print("=" * 60)

print()

print(
    "Total missing terrain cells:",
    len(missing_grid)
)


centroid_inside = (
    centroid_join["DISTRICT"].notna().sum()
)

centroid_outside = (
    centroid_join["DISTRICT"].isna().sum()
)

print()

print(
    "Centroids inside NCR:",
    centroid_inside
)

print(
    "Centroids outside NCR:",
    centroid_outside
)


cell_intersects = (
    cell_join["DISTRICT"].notna().sum()
)

print()

print(
    "Grid cells intersecting NCR:",
    cell_intersects
)


# ============================================================
# SAMPLE CENTROIDS OUTSIDE NCR
# ============================================================

outside_ids = centroid_join.loc[
    centroid_join["DISTRICT"].isna(),
    "grid_id"
]

outside_cells = missing_grid[
    missing_grid["grid_id"].isin(outside_ids)
]


print()
print("First 20 missing cells whose centroid is outside NCR:")

print(
    outside_cells[
        [
            "grid_id",
            "latitude",
            "longitude"
        ]
    ].head(20)
)


# ============================================================
# DISTRICT COUNTS FOR CENTROIDS
# ============================================================

print()
print("Centroid district distribution:")

print(
    centroid_join["DISTRICT"]
    .value_counts(dropna=False)
)


print()
print("=" * 60)
print("Investigation complete.")
print("=" * 60)