from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRID_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_1km_grid.geojson"
)

SUSCEPTIBILITY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_spatial_susceptibility.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_spatial_susceptibility_with_coordinates.csv"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "FloodSense AI - Adding Grid Coordinates"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Load grid
    # --------------------------------------------------------

    print()
    print(
        "Loading grid..."
    )

    grid = gpd.read_file(
        GRID_FILE
    )

    print(
        f"Grid cells: {len(grid):,}"
    )

    print(
        f"Original CRS: {grid.crs}"
    )


    # --------------------------------------------------------
    # Project to UTM Zone 43N
    #
    # We calculate centroids in a projected CRS instead of
    # EPSG:4326 so that GeoPandas does not produce the
    # geographic-CRS centroid warning.
    # --------------------------------------------------------

    grid_utm = grid.to_crs(
        "EPSG:32643"
    )


    # --------------------------------------------------------
    # Calculate centroids
    # --------------------------------------------------------

    centroids_utm = (
        grid_utm.geometry.centroid
    )


    # --------------------------------------------------------
    # Convert centroid points to WGS84
    # --------------------------------------------------------

    centroids_wgs84 = (
        gpd.GeoSeries(
            centroids_utm,
            crs="EPSG:32643",
        )
        .to_crs(
            "EPSG:4326"
        )
    )


    # --------------------------------------------------------
    # Create coordinate table
    # --------------------------------------------------------

    coordinates = pd.DataFrame(
        {
            "grid_id":
                grid["grid_id"],

            "latitude":
                centroids_wgs84.y,

            "longitude":
                centroids_wgs84.x,
        }
    )


    # --------------------------------------------------------
    # Load susceptibility
    # --------------------------------------------------------

    susceptibility = pd.read_csv(
        SUSCEPTIBILITY_FILE
    )

    print()
    print(
        f"Susceptibility rows: "
        f"{len(susceptibility):,}"
    )


    # --------------------------------------------------------
    # Merge coordinates with susceptibility
    # --------------------------------------------------------

    result = susceptibility.merge(
        coordinates,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )


    # --------------------------------------------------------
    # Quality check: row count
    # --------------------------------------------------------

    print()
    print(
        f"Final rows: {len(result):,}"
    )


    # --------------------------------------------------------
    # Quality check: missing coordinates
    # --------------------------------------------------------

    print()
    print(
        "Coordinate missing values:"
    )

    missing = (
        result[
            [
                "latitude",
                "longitude",
            ]
        ]
        .isna()
        .sum()
    )

    print(
        missing.to_string()
    )


    if missing.sum() != 0:

        raise ValueError(
            "Some grid cells do not have coordinates."
        )


    # --------------------------------------------------------
    # Coordinate ranges
    # --------------------------------------------------------

    print()
    print(
        "Coordinate ranges:"
    )

    print(
        f"Latitude : "
        f"{result['latitude'].min():.6f} "
        f"to "
        f"{result['latitude'].max():.6f}"
    )

    print(
        f"Longitude: "
        f"{result['longitude'].min():.6f} "
        f"to "
        f"{result['longitude'].max():.6f}"
    )


    # --------------------------------------------------------
    # Grid ID uniqueness
    # --------------------------------------------------------

    if result["grid_id"].duplicated().any():

        raise ValueError(
            "Duplicate grid_id values detected."
        )


    print()
    print(
        "Grid ID check: passed"
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )


    print()
    print("=" * 70)

    print(
        "Grid coordinates added successfully."
    )

    print("=" * 70)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()