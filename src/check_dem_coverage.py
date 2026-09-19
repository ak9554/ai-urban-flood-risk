import rasterio
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

DEM_FOLDER = Path("data/raw/elevation")


# ============================================================
# FIND DEM FILES
# ============================================================

dem_files = sorted(DEM_FOLDER.glob("*.tif"))

print("=" * 60)
print("COPERNICUS DEM COVERAGE CHECK")
print("=" * 60)

print("DEM files found:", len(dem_files))

if not dem_files:
    print("No DEM files found.")
    raise SystemExit


# ============================================================
# INSPECT EACH DEM
# ============================================================

for dem_file in dem_files:

    print()
    print("-" * 60)
    print("File:", dem_file.name)

    with rasterio.open(dem_file) as src:

        print("CRS:", src.crs)
        print("Width:", src.width)
        print("Height:", src.height)

        print("Bounds:")
        print("  Left:", src.bounds.left)
        print("  Bottom:", src.bounds.bottom)
        print("  Right:", src.bounds.right)
        print("  Top:", src.bounds.top)

        print("Resolution:")
        print("  X:", src.res[0])
        print("  Y:", src.res[1])

        print("NoData value:", src.nodata)

        # Read raster
        data = src.read(1)

        # Count valid and NoData pixels
        if src.nodata is not None:
            nodata_pixels = (data == src.nodata).sum()
        else:
            nodata_pixels = 0

        total_pixels = data.size
        valid_pixels = total_pixels - nodata_pixels

        print("Total pixels:", total_pixels)
        print("Valid pixels:", valid_pixels)
        print("NoData pixels:", nodata_pixels)

        print(
            "NoData percentage:",
            round(nodata_pixels / total_pixels * 100, 2),
            "%"
        )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("Coverage check complete.")
print("=" * 60)