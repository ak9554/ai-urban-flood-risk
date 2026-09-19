import rasterio
import geopandas as gpd
from rasterio.mask import mask
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

DEM_FOLDER = Path(
    "data/raw/elevation"
)

BOUNDARY_FILE = Path(
    "data/processed/ncr_7_districts.geojson"
)

OUTPUT_FOLDER = Path(
    "data/processed/elevation"
)


# ============================================================
# LOAD NCR BOUNDARY
# ============================================================

boundary = gpd.read_file(
    BOUNDARY_FILE
)

print(
    "NCR boundary CRS:",
    boundary.crs
)


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PROCESS EACH DEM
# ============================================================

dem_files = sorted(
    DEM_FOLDER.glob("*.tif")
)

print(
    "DEM files found:",
    len(dem_files)
)


for dem_file in dem_files:

    print()
    print("=" * 60)
    print(
        "Processing:",
        dem_file.name
    )


    with rasterio.open(dem_file) as src:

        # ----------------------------------------------------
        # Convert boundary to DEM CRS
        # ----------------------------------------------------

        boundary_for_dem = boundary.to_crs(
            src.crs
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # Include pixels touched by the boundary.
        # This prevents valid edge pixels from being removed.
        # ----------------------------------------------------

        shapes = (
            boundary_for_dem
            .geometry
            .values
        )


        # ----------------------------------------------------
        # Clip DEM
        # ----------------------------------------------------

        out_image, out_transform = mask(
            src,
            shapes,
            crop=True,
            nodata=-9999,
            filled=True,
            all_touched=True
        )


        # ----------------------------------------------------
        # Update metadata
        # ----------------------------------------------------

        out_meta = src.meta.copy()

        out_meta.update({

            "driver": "GTiff",

            "height": out_image.shape[1],

            "width": out_image.shape[2],

            "transform": out_transform,

            "nodata": -9999

        })


        # ----------------------------------------------------
        # Output filename
        # ----------------------------------------------------

        output_file = (
            OUTPUT_FOLDER
            / f"{dem_file.stem}_NCR_clip.tif"
        )


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        with rasterio.open(
            output_file,
            "w",
            **out_meta
        ) as dst:

            dst.write(
                out_image
            )


        print(
            "Output:",
            output_file
        )

        print(
            "Width:",
            out_image.shape[2]
        )

        print(
            "Height:",
            out_image.shape[1]
        )


print()
print("=" * 60)
print("DEM clipping complete.")
print("=" * 60)