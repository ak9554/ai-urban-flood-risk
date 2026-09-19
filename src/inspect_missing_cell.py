import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

GRID_FILE = Path("data/processed/ncr_1km_grid.geojson")

TARGET_GRID_ID = "NCR_0059"

RAW_DEM_FOLDER = Path("data/raw/elevation")
CLIPPED_DEM_FOLDER = Path("data/processed/elevation")


# ============================================================
# LOAD GRID
# ============================================================

grid = gpd.read_file(GRID_FILE)

cell = grid[
    grid["grid_id"] == TARGET_GRID_ID
].copy()

if cell.empty:
    print("Grid cell not found.")
    raise SystemExit


print("=" * 60)
print("MISSING GRID CELL INSPECTION")
print("=" * 60)

print()
print("Grid ID:", TARGET_GRID_ID)
print("Centroid latitude:", cell.iloc[0]["latitude"])
print("Centroid longitude:", cell.iloc[0]["longitude"])

print()
print("Grid geometry bounds:")
print(cell.total_bounds)


# ============================================================
# CHECK RAW DEM FILES
# ============================================================

print()
print("=" * 60)
print("RAW DEM CHECK")
print("=" * 60)


for dem_file in sorted(RAW_DEM_FOLDER.glob("*.tif")):

    with rasterio.open(dem_file) as src:

        print()
        print("DEM:", dem_file.name)

        print(
            "Bounds:",
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top
        )

        # Check whether grid cell intersects raster bounds
        intersects = (
            cell.geometry.iloc[0].bounds[2] >= src.bounds.left
            and cell.geometry.iloc[0].bounds[0] <= src.bounds.right
            and cell.geometry.iloc[0].bounds[3] >= src.bounds.bottom
            and cell.geometry.iloc[0].bounds[1] <= src.bounds.top
        )

        print("Grid cell intersects DEM:", intersects)

        if not intersects:
            continue

        # Try extracting the cell directly from the RAW DEM
        try:

            out_image, out_transform = mask(
                src,
                cell.geometry,
                crop=True,
                filled=False
            )

            data = out_image[0]

            valid = data.compressed()

            print("Pixels extracted:", data.size)
            print("Valid pixels:", len(valid))

            if len(valid) > 0:
                print("Minimum elevation:", valid.min())
                print("Maximum elevation:", valid.max())
                print("Mean elevation:", valid.mean())
            else:
                print("NO VALID PIXELS FROM RAW DEM")

        except Exception as e:

            print("Extraction error:", e)


# ============================================================
# CHECK CLIPPED DEM FILES
# ============================================================

print()
print("=" * 60)
print("CLIPPED DEM CHECK")
print("=" * 60)


for dem_file in sorted(CLIPPED_DEM_FOLDER.glob("*_NCR_clip.tif")):

    with rasterio.open(dem_file) as src:

        print()
        print("Clipped DEM:", dem_file.name)

        print(
            "Bounds:",
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top
        )

        try:

            out_image, out_transform = mask(
                src,
                cell.geometry,
                crop=True,
                filled=False
            )

            data = out_image[0]

            valid = data.compressed()

            print("Pixels extracted:", data.size)
            print("Valid pixels:", len(valid))

            if len(valid) > 0:
                print("Minimum elevation:", valid.min())
                print("Maximum elevation:", valid.max())
                print("Mean elevation:", valid.mean())
            else:
                print("NO VALID PIXELS FROM CLIPPED DEM")

        except Exception as e:

            print("Extraction error:", e)


print()
print("=" * 60)
print("Inspection complete.")
print("=" * 60)