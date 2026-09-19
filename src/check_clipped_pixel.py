import rasterio
from pathlib import Path


# ============================================================
# TEST LOCATION
# ============================================================

LATITUDE = 29.08270140473846
LONGITUDE = 76.51346581786152

RAW_DEM = Path(
    "data/raw/elevation/N29_E076_DEM.tif"
)

CLIPPED_DEM = Path(
    "data/processed/elevation/N29_E076_DEM_NCR_clip.tif"
)


# ============================================================
# FUNCTION TO CHECK ONE RASTER
# ============================================================

def check_raster(file_path):

    print()
    print("=" * 60)
    print("FILE:", file_path.name)
    print("=" * 60)

    with rasterio.open(file_path) as src:

        print("Bounds:", src.bounds)
        print("CRS:", src.crs)
        print("NoData:", src.nodata)

        # Convert coordinate to raster row/column
        row, col = src.index(
            LONGITUDE,
            LATITUDE
        )

        print()
        print("Raster row:", row)
        print("Raster column:", col)

        # Read exactly one pixel
        value = src.read(
            1,
            window=rasterio.windows.Window(
                col_off=col,
                row_off=row,
                width=1,
                height=1
            )
        )[0, 0]

        print()
        print("Pixel value:", value)

        # Check whether it is our NoData value
        if src.nodata is not None and value == src.nodata:
            print("RESULT: NoData pixel")
        else:
            print("RESULT: VALID PIXEL")


# ============================================================
# RUN CHECKS
# ============================================================

print("=" * 60)
print("CLIPPED DEM PIXEL CHECK")
print("=" * 60)

print()
print("Test coordinate:")
print("Latitude:", LATITUDE)
print("Longitude:", LONGITUDE)

check_raster(RAW_DEM)
check_raster(CLIPPED_DEM)

print()
print("=" * 60)
print("Check complete.")
print("=" * 60)