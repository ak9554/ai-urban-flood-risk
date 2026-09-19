import rasterio
import numpy as np
from pathlib import Path

LULC_FILE = Path(
    "data/raw/lulc/ESA_WorldCover_10m_2021_N27E075_Map.tif"
)

print("=" * 60)
print("ESA WorldCover inspection")
print("=" * 60)

with rasterio.open(LULC_FILE) as src:
    print("CRS:", src.crs)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Bands:", src.count)
    print("Resolution:", src.res)
    print("Bounds:", src.bounds)
    print("NoData:", src.nodata)
    print("Data type:", src.dtypes[0])

    data = src.read(1)

    valid = data[data != src.nodata] if src.nodata is not None else data

    print()
    print("Valid pixels:", len(valid))

    unique, counts = np.unique(valid, return_counts=True)

    print()
    print("Land-cover classes found:")
    for value, count in zip(unique, counts):
        percentage = (count / len(valid)) * 100
        print(
            f"Class {int(value):>3}: "
            f"{count:>12,} pixels "
            f"({percentage:.2f}%)"
        )

print()
print("=" * 60)
print("Inspection complete.")
print("=" * 60)