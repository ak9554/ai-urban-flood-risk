import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

GRID_FILE = Path("data/processed/ncr_1km_grid.geojson")

DEM_FILE = Path(
    "data/raw/elevation/N29_E076_DEM.tif"
)

TARGET_GRID_ID = "NCR_0059"


# ============================================================
# LOAD GRID CELL
# ============================================================

grid = gpd.read_file(GRID_FILE)

cell = grid[
    grid["grid_id"] == TARGET_GRID_ID
].iloc[0]

minx, miny, maxx, maxy = cell.geometry.bounds


print("=" * 60)
print("RAW DEM PIXEL INSPECTION")
print("=" * 60)

print()
print("Grid ID:", TARGET_GRID_ID)

print("Grid bounds:")
print("Left:", minx)
print("Bottom:", miny)
print("Right:", maxx)
print("Top:", maxy)


# ============================================================
# OPEN DEM
# ============================================================

with rasterio.open(DEM_FILE) as src:

    print()
    print("DEM:", DEM_FILE.name)

    print("DEM bounds:")
    print(src.bounds)

    print()
    print("DEM NoData:", src.nodata)

    # --------------------------------------------------------
    # Calculate raster window covering the grid cell
    # --------------------------------------------------------

    window = from_bounds(
        minx,
        miny,
        maxx,
        maxy,
        transform=src.transform
    )

    print()
    print("Raster window:")
    print(window)

    # Read pixels
    data = src.read(
        1,
        window=window
    )

    print()
    print("Raw pixel array shape:", data.shape)

    print()
    print("Raw pixel values:")

    print(data)

    print()
    print("Minimum pixel value:", data.min())
    print("Maximum pixel value:", data.max())
    print("Mean pixel value:", data.mean())

    print()
    print("Unique pixel values:")

    print(set(data.flatten()))


print()
print("=" * 60)
print("Inspection complete.")
print("=" * 60)