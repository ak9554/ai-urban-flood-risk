import geopandas as gpd
import pandas as pd
from pathlib import Path


GRID_FILE = Path(
    "data/processed/ncr_1km_grid.geojson"
)

BOUNDARY_FILE = Path(
    "data/processed/ncr_7_districts.geojson"
)

OUTPUT_FILE = Path(
    "data/processed/ncr_grid_districts.csv"
)


print("=" * 60)
print("Assigning 1-km grid cells to NCR districts")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load grid
# ------------------------------------------------------------

grid = gpd.read_file(GRID_FILE)

print()
print("Grid cells:", len(grid))

print("Grid CRS:", grid.crs)


# ------------------------------------------------------------
# 2. Load NCR boundaries
# ------------------------------------------------------------

districts = gpd.read_file(
    BOUNDARY_FILE
)

print()
print("District polygons:", len(districts))

print(
    "District CRS:",
    districts.crs
)


# ------------------------------------------------------------
# 3. Make CRS identical
# ------------------------------------------------------------

districts = districts.to_crs(
    grid.crs
)


# ------------------------------------------------------------
# 4. Spatial join
# ------------------------------------------------------------

grid_district = gpd.sjoin(
    grid[
        [
            "grid_id",
            "longitude",
            "latitude",
            "geometry"
        ]
    ],
    districts[
        [
            "DISTRICT",
            "geometry"
        ]
    ],
    how="left",
    predicate="intersects"
)


# ------------------------------------------------------------
# 5. Remove duplicate grid assignments
# ------------------------------------------------------------

grid_district = (
    grid_district
    .sort_values("grid_id")
    .drop_duplicates(
        subset=["grid_id"]
    )
)


# ------------------------------------------------------------
# 6. Rename district column
# ------------------------------------------------------------

grid_district = grid_district.rename(
    columns={
        "DISTRICT": "district"
    }
)


# ------------------------------------------------------------
# 7. Keep only required columns
# ------------------------------------------------------------

grid_district = grid_district[
    [
        "grid_id",
        "latitude",
        "longitude",
        "district"
    ]
]


# ------------------------------------------------------------
# 8. Check assignments
# ------------------------------------------------------------

missing = grid_district[
    grid_district["district"].isna()
]


print()
print("Grid cells assigned:", len(grid_district))

print(
    "Grid cells without district:",
    len(missing)
)


print()
print("Grid cells by district:")

print(
    grid_district["district"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# 9. Check duplicate grid IDs
# ------------------------------------------------------------

duplicates = (
    grid_district["grid_id"]
    .duplicated()
    .sum()
)


print()
print(
    "Duplicate grid IDs:",
    duplicates
)


# ------------------------------------------------------------
# 10. Save
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

grid_district.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("Saved:", OUTPUT_FILE)

print()
print("=" * 60)
print("Grid-district assignment complete.")
print("=" * 60)