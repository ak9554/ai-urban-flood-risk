import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pathlib import Path


GRID_FILE = Path(
    "data/processed/ncr_1km_grid.geojson"
)

DEM_FILE = Path(
    "data/raw/elevation/N29_E076_DEM.tif"
)

TARGET_GRID_ID = "NCR_0059"


# ============================================================
# LOAD GRID
# ============================================================

grid = gpd.read_file(GRID_FILE)

cell = grid[
    grid["grid_id"] == TARGET_GRID_ID
].copy()


print("=" * 60)
print("POLYGON EXTRACTION TEST")
print("=" * 60)

print()
print("Grid:", TARGET_GRID_ID)

print(
    "Grid bounds:",
    cell.total_bounds
)


# ============================================================
# OPEN DEM
# ============================================================

with rasterio.open(DEM_FILE) as src:

    print()
    print("DEM:", DEM_FILE.name)

    print("DEM CRS:", src.crs)

    # Make sure CRS matches
    cell = cell.to_crs(src.crs)


    # ========================================================
    # TEST 1 — NORMAL MASK
    # ========================================================

    print()
    print("-" * 60)
    print("TEST 1: all_touched=False")
    print("-" * 60)

    out_image, out_transform = mask(
        src,
        cell.geometry,
        crop=True,
        filled=False,
        all_touched=False
    )

    data = out_image[0]

    valid = data.compressed()

    print("Total extracted pixels:", data.size)
    print("Valid pixels:", len(valid))

    if len(valid) > 0:
        print("Minimum:", valid.min())
        print("Maximum:", valid.max())
        print("Mean:", valid.mean())
    else:
        print("NO VALID PIXELS")


    # ========================================================
    # TEST 2 — ALL TOUCHED
    # ========================================================

    print()
    print("-" * 60)
    print("TEST 2: all_touched=True")
    print("-" * 60)

    out_image, out_transform = mask(
        src,
        cell.geometry,
        crop=True,
        filled=False,
        all_touched=True
    )

    data = out_image[0]

    valid = data.compressed()

    print("Total extracted pixels:", data.size)
    print("Valid pixels:", len(valid))

    if len(valid) > 0:
        print("Minimum:", valid.min())
        print("Maximum:", valid.max())
        print("Mean:", valid.mean())
    else:
        print("NO VALID PIXELS")


print()
print("=" * 60)
print("Test complete.")
print("=" * 60)