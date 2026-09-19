import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
from pathlib import Path


GRID_FILE = Path("data/processed/ncr_1km_grid.geojson")
LULC_FILE = Path(
    "data/raw/lulc/ESA_WorldCover_10m_2021_N27E075_Map.tif"
)
OUTPUT_FILE = Path("data/processed/ncr_lulc_features.csv")


print("=" * 60)
print("NCR LULC feature extraction")
print("=" * 60)

# ------------------------------------------------------------
# 1. Load the 1-km grid
# ------------------------------------------------------------

grid = gpd.read_file(GRID_FILE)

print("Grid cells:", len(grid))
print("Grid CRS:", grid.crs)

# ------------------------------------------------------------
# 2. Open WorldCover raster
# ------------------------------------------------------------

with rasterio.open(LULC_FILE) as src:

    print("LULC CRS:", src.crs)
    print("LULC resolution:", src.res)

    # Reproject grid if necessary
    grid_for_raster = grid.to_crs(src.crs)

    results = []

    # --------------------------------------------------------
    # 3. Process each 1-km cell
    # --------------------------------------------------------

    for index, row in grid_for_raster.iterrows():

        grid_id = row["grid_id"]
        geometry = [row.geometry]

        try:
            data, _ = mask(
                src,
                geometry,
                crop=True,
                filled=False
            )

            values = data[0].compressed()

            if len(values) == 0:
                results.append({
                    "grid_id": grid_id,
                    "builtup_percent": np.nan,
                    "vegetation_percent": np.nan,
                    "water_percent": np.nan,
                    "cropland_percent": np.nan,
                    "bare_percent": np.nan
                })
                continue

            total = len(values)

            # WorldCover classes
            tree = np.sum(values == 10)
            shrub = np.sum(values == 20)
            grass = np.sum(values == 30)
            crop = np.sum(values == 40)
            builtup = np.sum(values == 50)
            bare = np.sum(values == 60)
            water = np.sum(values == 80)
            wetland = np.sum(values == 90)

            vegetation = tree + shrub + grass + wetland

            results.append({
                "grid_id": grid_id,

                "builtup_percent":
                    (builtup / total) * 100,

                "vegetation_percent":
                    (vegetation / total) * 100,

                "water_percent":
                    (water / total) * 100,

                "cropland_percent":
                    (crop / total) * 100,

                "bare_percent":
                    (bare / total) * 100
            })

        except Exception as e:

            print(
                f"Error processing {grid_id}: {e}"
            )

            results.append({
                "grid_id": grid_id,
                "builtup_percent": np.nan,
                "vegetation_percent": np.nan,
                "water_percent": np.nan,
                "cropland_percent": np.nan,
                "bare_percent": np.nan
            })

        # Progress every 500 cells
        if (index + 1) % 500 == 0:
            print(
                f"Processed {index + 1}/{len(grid)} cells"
            )


# ------------------------------------------------------------
# 4. Convert results to DataFrame
# ------------------------------------------------------------

lulc_features = pd.DataFrame(results)

print()
print("LULC feature rows:", len(lulc_features))

print()
print("Missing values:")
print(lulc_features.isna().sum())

print()
print("Feature statistics:")
print(lulc_features.describe())


# ------------------------------------------------------------
# 5. Save
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

lulc_features.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Saved:", OUTPUT_FILE)

print()
print("=" * 60)
print("LULC extraction complete.")
print("=" * 60)