import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask


GRID_FILE = "data/processed/ncr_1km_grid.geojson"
LULC_FILE = "data/raw/lulc/ESA_WorldCover_10m_2021_N27E075_Map.tif"

MISSING_IDS = [
    "NCR_6111",
    "NCR_6974",
    "NCR_8452"
]


print("=" * 60)
print("Checking missing LULC cells against raster")
print("=" * 60)


# ------------------------------------------------------------
# Load grid
# ------------------------------------------------------------

grid = gpd.read_file(GRID_FILE)

missing_grid = grid[
    grid["grid_id"].isin(MISSING_IDS)
].copy()

print("Cells being checked:", len(missing_grid))


# ------------------------------------------------------------
# Open raster
# ------------------------------------------------------------

with rasterio.open(LULC_FILE) as src:

    print("Raster CRS:", src.crs)
    print("Raster bounds:", src.bounds)

    missing_grid = missing_grid.to_crs(src.crs)

    # --------------------------------------------------------
    # Check each cell
    # --------------------------------------------------------

    for _, row in missing_grid.iterrows():

        grid_id = row["grid_id"]

        print()
        print("-" * 60)
        print("Grid:", grid_id)

        try:

            data, transform = mask(
                src,
                [row.geometry],
                crop=True,
                filled=False
            )

            values = data[0].compressed()

            print("Pixels returned:", len(values))

            if len(values) == 0:
                print("RESULT: No valid WorldCover pixels")

            else:
                print("RESULT: Valid pixels found")
                print("Unique classes:", set(values.tolist()))

        except Exception as e:

            print("ERROR:", e)


print()
print("=" * 60)
print("Raster coverage check complete.")
print("=" * 60)