import rasterio
from pathlib import Path

DEM_FOLDER = Path("data/raw/elevation")

dem_files = list(DEM_FOLDER.glob("*.tif"))

print("Number of DEM files:", len(dem_files))
print()

for dem_file in dem_files:
    print("=" * 60)
    print("File:", dem_file.name)

    with rasterio.open(dem_file) as src:
        print("CRS:", src.crs)
        print("Width:", src.width)
        print("Height:", src.height)
        print("Resolution:", src.res)
        print("Bounds:", src.bounds)
        print("NoData:", src.nodata)

        elevation = src.read(1, masked=True)

        print("Minimum elevation:", float(elevation.min()))
        print("Maximum elevation:", float(elevation.max()))
        print("Mean elevation:", float(elevation.mean()))

print()
print("DEM inspection complete.")