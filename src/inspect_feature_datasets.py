import pandas as pd
from pathlib import Path


TERRAIN_FILE = Path(
    "data/processed/ncr_terrain_features.csv"
)

LULC_FILE = Path(
    "data/processed/ncr_lulc_features.csv"
)

GRID_FILE = Path(
    "data/processed/ncr_1km_grid.geojson"
)


print("=" * 60)
print("Inspecting grid feature datasets")
print("=" * 60)


# ------------------------------------------------------------
# 1. Terrain
# ------------------------------------------------------------

terrain = pd.read_csv(TERRAIN_FILE)

print()
print("TERRAIN")
print("-" * 40)

print("Rows:", len(terrain))

print("Columns:")
print(
    terrain.columns.tolist()
)

print()
print("First rows:")

print(
    terrain.head()
    .to_string(index=False)
)


# ------------------------------------------------------------
# 2. LULC
# ------------------------------------------------------------

lulc = pd.read_csv(LULC_FILE)

print()
print("LULC")
print("-" * 40)

print("Rows:", len(lulc))

print("Columns:")
print(
    lulc.columns.tolist()
)

print()
print("First rows:")

print(
    lulc.head()
    .to_string(index=False)
)


# ------------------------------------------------------------
# 3. Grid
# ------------------------------------------------------------

import geopandas as gpd

grid = gpd.read_file(GRID_FILE)

print()
print("GRID")
print("-" * 40)

print("Rows:", len(grid))

print("Columns:")
print(
    grid.columns.tolist()
)

print()
print("CRS:", grid.crs)

print()
print("First rows:")

print(
    grid.head()
    .to_string(index=False)
)


# ------------------------------------------------------------
# 4. Grid ID consistency
# ------------------------------------------------------------

terrain_ids = set(
    terrain["grid_id"]
)

lulc_ids = set(
    lulc["grid_id"]
)

grid_ids = set(
    grid["grid_id"]
)


print()
print("GRID ID CONSISTENCY")
print("-" * 40)

print(
    "Terrain IDs:",
    len(terrain_ids)
)

print(
    "LULC IDs:",
    len(lulc_ids)
)

print(
    "Grid IDs:",
    len(grid_ids)
)


print(
    "Terrain missing from grid:",
    len(terrain_ids - grid_ids)
)

print(
    "LULC missing from grid:",
    len(lulc_ids - grid_ids)
)

print(
    "Grid missing from terrain:",
    len(grid_ids - terrain_ids)
)

print(
    "Grid missing from LULC:",
    len(grid_ids - lulc_ids)
)


# ------------------------------------------------------------
# 5. Missing values
# ------------------------------------------------------------

print()
print("MISSING VALUES")
print("-" * 40)

print("Terrain:")
print(
    terrain.isna()
    .sum()
    .to_string()
)

print()
print("LULC:")
print(
    lulc.isna()
    .sum()
    .to_string()
)


print()
print("=" * 60)
print("Feature dataset inspection complete.")
print("=" * 60)