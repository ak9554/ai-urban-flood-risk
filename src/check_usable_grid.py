import pandas as pd
from pathlib import Path


GRID_DISTRICT_FILE = Path(
    "data/processed/ncr_grid_districts.csv"
)

LULC_FILE = Path(
    "data/processed/ncr_lulc_features.csv"
)


print("=" * 60)
print("Checking usable grid cells")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load files
# ------------------------------------------------------------

grid = pd.read_csv(
    GRID_DISTRICT_FILE
)

lulc = pd.read_csv(
    LULC_FILE
)


# ------------------------------------------------------------
# 2. Identify complete LULC cells
# ------------------------------------------------------------

lulc_features = [
    "builtup_percent",
    "vegetation_percent",
    "water_percent",
    "cropland_percent",
    "bare_percent"
]


lulc["lulc_complete"] = (
    lulc[lulc_features]
    .notna()
    .all(axis=1)
)


# ------------------------------------------------------------
# 3. Keep usable cells
# ------------------------------------------------------------

usable_lulc = lulc[
    lulc["lulc_complete"]
][
    ["grid_id"]
]


usable_grid = grid.merge(
    usable_lulc,
    on="grid_id",
    how="inner"
)


# ------------------------------------------------------------
# 4. Results
# ------------------------------------------------------------

print()
print("Total grid cells:", len(grid))

print(
    "Grid cells with complete LULC:",
    len(usable_grid)
)

print(
    "Excluded cells:",
    len(grid) - len(usable_grid)
)


print()
print("Usable cells by district:")

print(
    usable_grid["district"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# 5. Check duplicates
# ------------------------------------------------------------

duplicates = (
    usable_grid["grid_id"]
    .duplicated()
    .sum()
)


print()
print(
    "Duplicate grid IDs:",
    duplicates
)


print()
print("=" * 60)
print("Usable grid check complete.")
print("=" * 60)