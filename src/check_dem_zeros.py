import rasterio
import numpy as np
from pathlib import Path


DEM_FOLDER = Path("data/processed/elevation")

dem_files = sorted(DEM_FOLDER.glob("*_NCR_clip.tif"))

for dem_file in dem_files:

    print("=" * 60)
    print("File:", dem_file.name)

    with rasterio.open(dem_file) as src:

        elevation = src.read(1)

        total_pixels = elevation.size
        zero_pixels = np.count_nonzero(elevation == 0)
        positive_pixels = np.count_nonzero(elevation > 0)

        print("Total pixels:", total_pixels)
        print("Zero-elevation pixels:", zero_pixels)
        print("Positive-elevation pixels:", positive_pixels)

        zero_percentage = (zero_pixels / total_pixels) * 100

        print("Zero-elevation percentage:", round(zero_percentage, 4), "%")

print()
print("Zero-value check complete.")