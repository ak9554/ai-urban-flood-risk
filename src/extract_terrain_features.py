import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

GRID_FILE = Path(
    "data/processed/ncr_1km_grid.geojson"
)

ELEVATION_FOLDER = Path(
    "data/processed/elevation"
)

OUTPUT_FILE = Path(
    "data/processed/ncr_terrain_features.csv"
)


# ============================================================
# LOAD GRID
# ============================================================

grid = gpd.read_file(GRID_FILE)

print("Number of grid cells:", len(grid))
print("Grid CRS:", grid.crs)


# ============================================================
# FIND RASTERS
# ============================================================

elevation_files = sorted(
    ELEVATION_FOLDER.glob("*_NCR_clip.tif")
)

slope_files = sorted(
    ELEVATION_FOLDER.glob("*_slope.tif")
)

print("Elevation files:", len(elevation_files))
print("Slope files:", len(slope_files))


# ============================================================
# EXTRACT RASTER VALUES
# ============================================================

def extract_raster_values(
    geometry,
    raster_files
):

    values = []

    for raster_file in raster_files:

        with rasterio.open(raster_file) as src:

            # ------------------------------------------------
            # Make sure geometry uses raster CRS
            # ------------------------------------------------

            geometry_gdf = gpd.GeoDataFrame(
                geometry=[geometry],
                crs=grid.crs
            ).to_crs(src.crs)

            raster_geometry = geometry_gdf.geometry.iloc[0]


            # ------------------------------------------------
            # Check whether geometry overlaps raster
            # ------------------------------------------------

            minx, miny, maxx, maxy = (
                raster_geometry.bounds
            )

            if (
                maxx < src.bounds.left
                or minx > src.bounds.right
                or maxy < src.bounds.bottom
                or miny > src.bounds.top
            ):
                continue


            # ------------------------------------------------
            # Extract pixels touched by the grid cell
            # ------------------------------------------------

            out_image, out_transform = mask(
                src,
                [raster_geometry],
                crop=True,
                filled=False,
                all_touched=True
            )


            data = out_image[0]


            # ------------------------------------------------
            # Get valid pixels
            # ------------------------------------------------

            valid_values = data.compressed()


            if len(valid_values) > 0:

                values.extend(
                    valid_values.astype(float)
                )


    return np.asarray(
        values,
        dtype=float
    )


# ============================================================
# PROCESS GRID
# ============================================================

results = []


for index, row in grid.iterrows():

    if index % 500 == 0:

        print(
            f"Processing grid cell "
            f"{index + 1}/{len(grid)}"
        )


    geometry = row.geometry


    # ========================================================
    # ELEVATION
    # ========================================================

    elevation_values = extract_raster_values(
        geometry,
        elevation_files
    )


    if elevation_values.size > 0:

        elevation_mean = elevation_values.mean()
        elevation_min = elevation_values.min()
        elevation_max = elevation_values.max()

    else:

        elevation_mean = np.nan
        elevation_min = np.nan
        elevation_max = np.nan


    # ========================================================
    # SLOPE
    # ========================================================

    slope_values = extract_raster_values(
        geometry,
        slope_files
    )


    if slope_values.size > 0:

        slope_mean = slope_values.mean()
        slope_min = slope_values.min()
        slope_max = slope_values.max()

    else:

        slope_mean = np.nan
        slope_min = np.nan
        slope_max = np.nan


    # ========================================================
    # SAVE RESULT
    # ========================================================

    results.append({

        "grid_id": row["grid_id"],

        "latitude": row["latitude"],

        "longitude": row["longitude"],

        "elevation_mean": elevation_mean,

        "elevation_min": elevation_min,

        "elevation_max": elevation_max,

        "slope_mean": slope_mean,

        "slope_min": slope_min,

        "slope_max": slope_max

    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

terrain = pd.DataFrame(results)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("Terrain extraction complete.")
print("=" * 60)

print("Rows:", len(terrain))

print("Columns:", list(terrain.columns))


print()
print("Missing values:")

print(
    terrain.isna().sum()
)


# ============================================================
# TERRAIN SUMMARY
# ============================================================

print()
print("Terrain summary:")

print(
    terrain[
        [
            "elevation_mean",
            "slope_mean"
        ]
    ].describe()
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

terrain.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Saved:", OUTPUT_FILE)