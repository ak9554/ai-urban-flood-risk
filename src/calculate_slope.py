import rasterio
import numpy as np
from pathlib import Path


DEM_FOLDER = Path("data/processed/elevation")
OUTPUT_FOLDER = Path("data/processed/elevation")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


dem_files = sorted(DEM_FOLDER.glob("*_NCR_clip.tif"))


for dem_file in dem_files:

    print()
    print("=" * 60)
    print("Processing:", dem_file.name)

    with rasterio.open(dem_file) as src:

        elevation = src.read(1).astype(float)

        # Identify NoData pixels
        nodata_mask = elevation == -9999

        # Replace NoData with the nearest valid elevation
        # temporarily so the gradient calculation does not
        # produce NaN values.
        valid_values = elevation[~nodata_mask]

        fill_value = float(np.mean(valid_values))

        elevation_for_gradient = elevation.copy()
        elevation_for_gradient[nodata_mask] = fill_value

        # Pixel size in degrees
        x_resolution = src.res[0]
        y_resolution = src.res[1]

        # Convert degrees to metres approximately
        latitude = (src.bounds.top + src.bounds.bottom) / 2

        meters_per_degree_lat = 111320
        meters_per_degree_lon = (
            111320 * np.cos(np.radians(latitude))
        )

        x_resolution_m = (
            x_resolution * meters_per_degree_lon
        )

        y_resolution_m = (
            y_resolution * meters_per_degree_lat
        )

        print("Pixel size (metres):")
        print("  X:", x_resolution_m)
        print("  Y:", y_resolution_m)

        # Calculate elevation gradients
        dz_dy, dz_dx = np.gradient(
            elevation_for_gradient,
            y_resolution_m,
            x_resolution_m
        )

        # Calculate slope in degrees
        slope_radians = np.arctan(
            np.sqrt(dz_dx ** 2 + dz_dy ** 2)
        )

        slope_degrees = np.degrees(slope_radians)

        # Mark the original NoData pixels again
        slope_degrees[nodata_mask] = -9999

        # Save slope raster
        output_file = (
            OUTPUT_FOLDER
            / f"{dem_file.stem.replace('_NCR_clip', '')}_slope.tif"
        )

        output_meta = src.meta.copy()

        output_meta.update({
            "dtype": "float32",
            "nodata": -9999,
            "count": 1
        })

        with rasterio.open(
            output_file,
            "w",
            **output_meta
        ) as dst:

            dst.write(
                slope_degrees.astype("float32"),
                1
            )

        # Only calculate statistics on valid pixels
        valid_slope = slope_degrees[
            slope_degrees != -9999
        ]

        print(
            "Minimum slope:",
            float(np.min(valid_slope)),
            "degrees"
        )

        print(
            "Maximum slope:",
            float(np.max(valid_slope)),
            "degrees"
        )

        print(
            "Mean slope:",
            float(np.mean(valid_slope)),
            "degrees"
        )

        print("Saved:", output_file)


print()
print("Slope calculation complete.")