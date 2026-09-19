import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pathlib import Path


GRID_FILE = Path(
    "data/processed/ncr_1km_grid.geojson"
)

RAW_DEM_FOLDER = Path(
    "data/raw/elevation"
)

CLIPPED_DEM_FOLDER = Path(
    "data/processed/elevation"
)

TARGET_GRID_ID = "NCR_2383"


# ============================================================
# LOAD GRID CELL
# ============================================================

grid = gpd.read_file(GRID_FILE)

cell = grid[
    grid["grid_id"] == TARGET_GRID_ID
].copy()

if cell.empty:
    print("Grid cell not found.")
    raise SystemExit


print("=" * 60)
print("REMAINING MISSING CELL INSPECTION")
print("=" * 60)

print()
print("Grid ID:", TARGET_GRID_ID)
print("Latitude:", cell.iloc[0]["latitude"])
print("Longitude:", cell.iloc[0]["longitude"])

print()
print("Grid bounds:")
print(cell.total_bounds)


# ============================================================
# CHECK ALL RAW DEM FILES
# ============================================================

print()
print("=" * 60)
print("RAW DEM CHECK")
print("=" * 60)


for dem_file in sorted(
    RAW_DEM_FOLDER.glob("*.tif")
):

    with rasterio.open(dem_file) as src:

        print()
        print("DEM:", dem_file.name)

        # Check overlap
        cell_bounds = cell.total_bounds

        overlaps = not (
            cell_bounds[2] < src.bounds.left
            or cell_bounds[0] > src.bounds.right
            or cell_bounds[3] < src.bounds.bottom
            or cell_bounds[1] > src.bounds.top
        )

        print("Grid intersects DEM:", overlaps)

        if not overlaps:
            continue


        # ----------------------------------------------------
        # TEST NORMAL MASK
        # ----------------------------------------------------

        print()
        print("all_touched=False:")

        try:

            out_image, _ = mask(
                src,
                cell.geometry,
                crop=True,
                filled=False,
                all_touched=False
            )

            valid = out_image[0].compressed()

            print(
                "Valid pixels:",
                len(valid)
            )

            if len(valid) > 0:
                print(
                    "Mean:",
                    valid.mean()
                )

        except Exception as e:

            print("Error:", e)


        # ----------------------------------------------------
        # TEST ALL TOUCHED
        # ----------------------------------------------------

        print()
        print("all_touched=True:")

        try:

            out_image, _ = mask(
                src,
                cell.geometry,
                crop=True,
                filled=False,
                all_touched=True
            )

            valid = out_image[0].compressed()

            print(
                "Valid pixels:",
                len(valid)
            )

            if len(valid) > 0:

                print(
                    "Minimum:",
                    valid.min()
                )

                print(
                    "Maximum:",
                    valid.max()
                )

                print(
                    "Mean:",
                    valid.mean()
                )

        except Exception as e:

            print("Error:", e)


# ============================================================
# CHECK CLIPPED DEM FILES
# ============================================================

print()
print("=" * 60)
print("CLIPPED DEM CHECK")
print("=" * 60)


for dem_file in sorted(
    CLIPPED_DEM_FOLDER.glob(
        "*_NCR_clip.tif"
    )
):

    with rasterio.open(dem_file) as src:

        print()
        print(
            "Clipped DEM:",
            dem_file.name
        )

        try:

            out_image, _ = mask(
                src,
                cell.geometry,
                crop=True,
                filled=False,
                all_touched=True
            )

            valid = out_image[0].compressed()

            print(
                "Valid pixels:",
                len(valid)
            )

            if len(valid) > 0:

                print(
                    "Minimum:",
                    valid.min()
                )

                print(
                    "Maximum:",
                    valid.max()
                )

                print(
                    "Mean:",
                    valid.mean()
                )

        except Exception as e:

            print("Error:", e)


print()
print("=" * 60)
print("Inspection complete.")
print("=" * 60)