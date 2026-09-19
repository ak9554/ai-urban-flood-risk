from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TERRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_terrain_features.csv"
)

LULC_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_lulc_features.csv"
)

DISTRICT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_grid_districts.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_spatial_susceptibility.csv"
)


# ============================================================
# Prototype weights
# ============================================================

WEIGHT_ELEVATION = 0.40
WEIGHT_SLOPE = 0.15
WEIGHT_BUILTUP = 0.30
WEIGHT_BARE = 0.05
WEIGHT_VEGETATION = 0.10


# ============================================================
# Min-max normalization
# ============================================================

def min_max_normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            0.5,
            index=series.index,
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


# ============================================================
# Load data
# ============================================================

def load_data():

    terrain = pd.read_csv(
        TERRAIN_FILE
    )

    lulc = pd.read_csv(
        LULC_FILE
    )

    districts = pd.read_csv(
        DISTRICT_FILE
    )

    print(
        f"Terrain rows: {len(terrain):,}"
    )

    print(
        f"LULC rows: {len(lulc):,}"
    )

    print(
        f"District rows: {len(districts):,}"
    )

    return (
        terrain,
        lulc,
        districts,
    )


# ============================================================
# Build susceptibility
# ============================================================

def build_susceptibility(
    terrain,
    lulc,
    districts,
):

    # --------------------------------------------------------
    # Merge spatial datasets
    # --------------------------------------------------------

    data = terrain.merge(
        lulc,
        on="grid_id",
        how="inner",
        suffixes=(
            "_terrain",
            "_lulc",
        ),
    )

    data = data.merge(
        districts,
        on="grid_id",
        how="left",
    )

    print()
    print(
        f"Merged grid cells: {len(data):,}"
    )


    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "grid_id",
        "elevation_mean",
        "slope_mean",
        "builtup_percent",
        "vegetation_percent",
        "water_percent",
        "cropland_percent",
        "bare_percent",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + str(missing_columns)
        )


    # ========================================================
    # Remove incomplete LULC cells
    # ========================================================

    lulc_columns = [
        "builtup_percent",
        "vegetation_percent",
        "water_percent",
        "cropland_percent",
        "bare_percent",
    ]

    before_count = len(data)

    data = data.dropna(
        subset=lulc_columns
    ).copy()

    removed_count = (
        before_count
        - len(data)
    )

    print()
    print(
        f"Removed incomplete LULC cells: "
        f"{removed_count}"
    )

    print(
        f"Usable grid cells: "
        f"{len(data):,}"
    )


    # ========================================================
    # Elevation susceptibility
    # ========================================================

    elevation_normalized = (
        min_max_normalize(
            data["elevation_mean"]
        )
    )

    data["elevation_susceptibility"] = (
        1
        - elevation_normalized
    )


    # ========================================================
    # Slope susceptibility
    # ========================================================

    slope_cap = (
        data["slope_mean"]
        .quantile(0.95)
    )

    slope_capped = (
        data["slope_mean"]
        .clip(
            upper=slope_cap
        )
    )

    data["slope_susceptibility"] = (
        min_max_normalize(
            slope_capped
        )
    )


    # ========================================================
    # Built-up susceptibility
    # ========================================================

    data["builtup_susceptibility"] = (
        data["builtup_percent"]
        / 100.0
    )


    # ========================================================
    # Bare land susceptibility
    # ========================================================

    data["bare_susceptibility"] = (
        data["bare_percent"]
        / 100.0
    )


    # ========================================================
    # Vegetation protection
    # ========================================================

    data["vegetation_protection"] = (
        data["vegetation_percent"]
        / 100.0
    )


    # ========================================================
    # Prototype raw susceptibility
    # ========================================================

    data["susceptibility_raw"] = (

        WEIGHT_ELEVATION
        * data["elevation_susceptibility"]

        +

        WEIGHT_SLOPE
        * data["slope_susceptibility"]

        +

        WEIGHT_BUILTUP
        * data["builtup_susceptibility"]

        +

        WEIGHT_BARE
        * data["bare_susceptibility"]

        -

        WEIGHT_VEGETATION
        * data["vegetation_protection"]
    )


    # ========================================================
    # Convert to 0-100
    # ========================================================

    normalized_score = (
        min_max_normalize(
            data["susceptibility_raw"]
        )
    )

    data["susceptibility_score"] = (
        normalized_score * 100
    )


    # ========================================================
    # Categories
    # ========================================================

    data["susceptibility_category"] = pd.cut(
        data["susceptibility_score"],
        bins=[
            -np.inf,
            20,
            40,
            60,
            80,
            np.inf,
        ],
        labels=[
            "Very Low",
            "Low",
            "Moderate",
            "High",
            "Very High",
        ],
    )


    # ========================================================
    # Dominant driver
    # ========================================================

    driver_columns = [
        "elevation_susceptibility",
        "slope_susceptibility",
        "builtup_susceptibility",
        "bare_susceptibility",
        "vegetation_protection",
    ]

    data["dominant_driver"] = (
        data[
            driver_columns
        ]
        .idxmax(
            axis=1
        )
    )


    driver_names = {

        "elevation_susceptibility":
            "Lower elevation",

        "slope_susceptibility":
            "Slope/runoff",

        "builtup_susceptibility":
            "Built-up land",

        "bare_susceptibility":
            "Bare land",

        "vegetation_protection":
            "Vegetation",
    }

    data["dominant_driver"] = (
        data["dominant_driver"]
        .map(
            driver_names
        )
    )


    # ========================================================
    # Final columns
    # ========================================================

    output_columns = [

        "grid_id",

        "district",

        "elevation_mean",

        "slope_mean",

        "builtup_percent",

        "vegetation_percent",

        "water_percent",

        "cropland_percent",

        "bare_percent",

        "elevation_susceptibility",

        "slope_susceptibility",

        "builtup_susceptibility",

        "bare_susceptibility",

        "vegetation_protection",

        "susceptibility_raw",

        "susceptibility_score",

        "susceptibility_category",

        "dominant_driver",
    ]


    result = data[
        output_columns
    ].copy()


    return result


# ============================================================
# Quality checks
# ============================================================

def quality_checks(result):

    print()
    print("=" * 70)

    print(
        "SPATIAL SUSCEPTIBILITY QUALITY CHECK"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Basic count
    # --------------------------------------------------------

    print()
    print(
        f"Final grid cells: "
        f"{len(result):,}"
    )


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print()
    print(
        "Missing values:"
    )

    missing = (
        result.isna()
        .sum()
    )

    missing = missing[
        missing > 0
    ]

    if len(missing) == 0:

        print(
            "No missing values."
        )

    else:

        print(
            missing.to_string()
        )


    # --------------------------------------------------------
    # Score statistics
    # --------------------------------------------------------

    print()
    print(
        "Susceptibility score statistics:"
    )

    print(
        result[
            "susceptibility_score"
        ]
        .describe()
        .round(3)
        .to_string()
    )


    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    print()
    print(
        "Susceptibility categories:"
    )

    print(
        result[
            "susceptibility_category"
        ]
        .value_counts(
            sort=False
        )
        .to_string()
    )


    # --------------------------------------------------------
    # District summary
    # --------------------------------------------------------

    print()
    print(
        "District susceptibility summary:"
    )

    district_summary = (
        result
        .groupby(
            "district",
            observed=True,
        )[
            "susceptibility_score"
        ]
        .agg(
            [
                "count",
                "mean",
                "min",
                "max",
            ]
        )
        .round(2)
    )

    print(
        district_summary
        .to_string()
    )


    # --------------------------------------------------------
    # Dominant drivers
    # --------------------------------------------------------

    print()
    print(
        "Dominant spatial drivers:"
    )

    print(
        result[
            "dominant_driver"
        ]
        .value_counts()
        .to_string()
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "FloodSense AI - Spatial Susceptibility Builder"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        terrain,
        lulc,
        districts,
    ) = load_data()


    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    result = build_susceptibility(
        terrain,
        lulc,
        districts,
    )


    # --------------------------------------------------------
    # Quality checks
    # --------------------------------------------------------

    quality_checks(
        result
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
        "Saved spatial susceptibility:"
    )

    print(
        OUTPUT_FILE
    )


    print()
    print("=" * 70)

    print(
        "Spatial susceptibility completed."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()