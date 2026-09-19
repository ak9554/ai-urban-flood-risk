import geopandas as gpd
import pandas as pd
import rasterio
from pathlib import Path


# ============================================================
# FILES
# ============================================================

GRID_FILE = Path("data/processed/ncr_1km_grid.geojson")
TERRAIN_FILE = Path("data/processed/ncr_terrain_features.csv")
DEM_FOLDER = Path("data/raw/elevation")


# ============================================================
# LOAD GRID + TERRAIN
# ============================================================

grid = gpd.read_file(GRID_FILE)
terrain = pd.read_csv(TERRAIN_FILE)

missing_ids = terrain.loc[
    terrain["elevation_mean"].isna(),
    "grid_id"
]

missing_grid = grid[
    grid["grid_id"].isin(missing_ids)
].copy()


# ============================================================
# GET COMBINED DEM COVERAGE
# ============================================================

dem_files = sorted(DEM_FOLDER.glob("*.tif"))

bounds = []

for dem_file in dem_files:

    with rasterio.open(dem_file) as src:

        bounds.append({
            "file": dem_file.name,
            "left": src.bounds.left,
            "right": src.bounds.right,
            "bottom": src.bounds.bottom,
            "top": src.bounds.top
        })


# Overall coverage
overall_left = min(b["left"] for b in bounds)
overall_right = max(b["right"] for b in bounds)
overall_bottom = min(b["bottom"] for b in bounds)
overall_top = max(b["top"] for b in bounds)


# ============================================================
# CHECK MISSING CELL CENTROIDS
# ============================================================

missing_grid["inside_dem_coverage"] = (
    (missing_grid["longitude"] >= overall_left)
    & (missing_grid["longitude"] <= overall_right)
    & (missing_grid["latitude"] >= overall_bottom)
    & (missing_grid["latitude"] <= overall_top)
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 60)
print("MISSING TERRAIN vs DEM COVERAGE")
print("=" * 60)

print()
print("DEM files:")

for b in bounds:
    print(
        f"{b['file']}: "
        f"Lon {b['left']:.6f} to {b['right']:.6f}, "
        f"Lat {b['bottom']:.6f} to {b['top']:.6f}"
    )

print()
print("Combined DEM coverage:")
print("Longitude:", overall_left, "to", overall_right)
print("Latitude:", overall_bottom, "to", overall_top)

print()
print("Total missing terrain cells:", len(missing_grid))

inside = missing_grid["inside_dem_coverage"].sum()
outside = (~missing_grid["inside_dem_coverage"]).sum()

print("Missing cells INSIDE DEM coverage:", inside)
print("Missing cells OUTSIDE DEM coverage:", outside)

print()

print(
    "Inside percentage:",
    round(inside / len(missing_grid) * 100, 2),
    "%"
)

print(
    "Outside percentage:",
    round(outside / len(missing_grid) * 100, 2),
    "%"
)


# ============================================================
# SHOW CELLS OUTSIDE COVERAGE
# ============================================================

outside_cells = missing_grid[
    ~missing_grid["inside_dem_coverage"]
]

print()
print("First 20 cells outside DEM coverage:")

print(
    outside_cells[
        [
            "grid_id",
            "latitude",
            "longitude"
        ]
    ].head(20)
)


print()
print("=" * 60)
print("Coverage comparison complete.")
print("=" * 60)