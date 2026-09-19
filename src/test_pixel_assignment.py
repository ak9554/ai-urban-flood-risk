import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy
from pathlib import Path


GRID_FILE = Path(
    "data/processed/ncr_1km_grid.geojson"
)

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


print("=" * 60)
print("PIXEL ASSIGNMENT TEST")
print("=" * 60)

print()
print("Grid:", TARGET_GRID_ID)

print(
    "Grid bounds:",
    cell.geometry.bounds
)


# ============================================================
# OPEN DEM
# ============================================================

with rasterio.open(DEM_FILE) as src:

    minx, miny, maxx, maxy = cell.geometry.bounds

    # Convert geographic coordinates to raster indices
    left_col, top_row = src.index(minx, maxy)
    right_col, bottom_row = src.index(maxx, miny)

    print()
    print("Raster indices:")
    print("Left column:", left_col)
    print("Right column:", right_col)
    print("Top row:", top_row)
    print("Bottom row:", bottom_row)


    # ========================================================
    # CORRECT INDEX ORDER
    # ========================================================

    row_start = min(top_row, bottom_row)
    row_end = max(top_row, bottom_row)

    col_start = min(left_col, right_col)
    col_end = max(left_col, right_col)


    print()
    print("Reading pixels:")
    print(
        "Rows:",
        row_start,
        "to",
        row_end
    )

    print(
        "Columns:",
        col_start,
        "to",
        col_end
    )


    # ========================================================
    # CREATE RASTER WINDOW
    # ========================================================

    width = col_end - col_start + 1
    height = row_end - row_start + 1

    window = Window(
        col_off=col_start,
        row_off=row_start,
        width=width,
        height=height
    )


    # ========================================================
    # READ PIXELS
    # ========================================================

    data = src.read(
        1,
        window=window
    )


    print()
    print("Array shape:", data.shape)

    print()
    print("Pixel values:")

    print(data)


    # ========================================================
    # PRINT PIXEL CENTERS
    # ========================================================

    print()
    print("Pixel centers:")

    for r in range(
        row_start,
        row_end + 1
    ):

        for c in range(
            col_start,
            col_end + 1
        ):

            lon, lat = xy(
                src.transform,
                r,
                c,
                offset="center"
            )

            value = src.read(
                1,
                window=Window(
                    col_off=c,
                    row_off=r,
                    width=1,
                    height=1
                )
            )[0, 0]

            print(
                f"row={r}, "
                f"col={c}, "
                f"lat={lat:.8f}, "
                f"lon={lon:.8f}, "
                f"value={value}"
            )


print()
print("=" * 60)
print("Test complete.")
print("=" * 60)