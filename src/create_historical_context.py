from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FLOODED_AREA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "District_FloodedArea.csv"
)

FLOOD_IMPACT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "District_FloodImpact.csv"
)

FLOOD_LABEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_flood_labels_2015_2023.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_historical_context.csv"
)


# ============================================================
# Selected NCR districts
# ============================================================

TARGET_DISTRICTS = [
    "FARIDABAD",
    "GAUTAM BUDDHA NAGAR",
    "GHAZIABAD",
    "GURUGRAM",
    "NEW DELHI",
    "PANIPAT",
    "SONIPAT",
]


# ============================================================
# Helper
# ============================================================

def clean_district_name(series):

    return (
        series
        .astype(str)
        .str.upper()
        .str.strip()
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "FloodSense AI - Historical Context Builder"
    )

    print("=" * 70)


    # ========================================================
    # Load flooded-area data
    # ========================================================

    print()
    print(
        "Loading historical flooded-area data..."
    )

    flooded_area = pd.read_csv(
        FLOODED_AREA_FILE
    )


    flooded_area["district"] = (
        clean_district_name(
            flooded_area["Dist_Name"]
        )
    )


    print(
        f"Flooded-area rows: "
        f"{len(flooded_area):,}"
    )


    # ========================================================
    # Load flood-impact data
    # ========================================================

    print()
    print(
        "Loading historical flood-impact data..."
    )

    impact = pd.read_csv(
        FLOOD_IMPACT_FILE
    )


    impact["district"] = (
        clean_district_name(
            impact["Dist_Name"]
        )
    )


    print(
        f"Flood-impact rows: "
        f"{len(impact):,}"
    )


    # ========================================================
    # Load district flood labels
    # ========================================================

    print()
    print(
        "Loading 2015-2023 flood labels..."
    )

    labels = pd.read_csv(
        FLOOD_LABEL_FILE,
        parse_dates=["date"],
    )


    labels["district"] = (
        clean_district_name(
            labels["district"]
        )
    )


    print(
        f"Flood-label rows: "
        f"{len(labels):,}"
    )


    # ========================================================
    # Filter selected districts
    # ========================================================

    flooded_area = flooded_area[
        flooded_area["district"].isin(
            TARGET_DISTRICTS
        )
    ].copy()


    impact = impact[
        impact["district"].isin(
            TARGET_DISTRICTS
        )
    ].copy()


    labels = labels[
        labels["district"].isin(
            TARGET_DISTRICTS
        )
    ].copy()


    # ========================================================
    # Select flooded-area columns
    # ========================================================

    area_columns = [
        "district",
        "Percent_Flooded_Area",
        "Permanent_Water",
        "Corrected_Percent_Flooded_Area",
    ]


    area_columns = [
        column
        for column in area_columns
        if column in flooded_area.columns
    ]


    area = flooded_area[
        area_columns
    ].copy()


    area = area.rename(
        columns={
            "Percent_Flooded_Area":
                "historical_flooded_area_percent",

            "Corrected_Percent_Flooded_Area":
                "corrected_flooded_area_percent",

            "Permanent_Water":
                "permanent_water_percent",
        }
    )


    # ========================================================
    # Select impact columns
    # ========================================================

    impact_columns = [
        "district",
        "Human_fatality",
        "Human_injured",
        "Population",
        "Mean_Flood_Duration",
    ]


    impact_columns = [
        column
        for column in impact_columns
        if column in impact.columns
    ]


    impact_clean = impact[
        impact_columns
    ].copy()


    impact_clean = impact_clean.rename(
        columns={
            "Human_fatality":
                "historical_fatalities",

            "Human_injured":
                "historical_injured",

            "Population":
                "population",

            "Mean_Flood_Duration":
                "historical_mean_flood_duration",
        }
    )


    # ========================================================
    # Calculate flood frequency
    # ========================================================

    flood_frequency = (
        labels
        .groupby(
            "district"
        )
        .size()
        .reset_index(
            name="historical_flood_days_2015_2023"
        )
    )


    # ========================================================
    # Merge context
    # ========================================================

    context = pd.DataFrame(
        {
            "district":
                TARGET_DISTRICTS
        }
    )


    context = context.merge(
        area,
        on="district",
        how="left",
    )


    context = context.merge(
        impact_clean,
        on="district",
        how="left",
    )


    context = context.merge(
        flood_frequency,
        on="district",
        how="left",
    )


    # ========================================================
    # Missing flood-frequency values
    #
    # PANIPAT has zero flood-label days in 2015-2023.
    # That is a real zero, not missing data.
    # ========================================================

    context[
        "historical_flood_days_2015_2023"
    ] = (
        context[
            "historical_flood_days_2015_2023"
        ]
        .fillna(0)
    )


    # ========================================================
    # Numeric conversion
    # ========================================================

    numeric_columns = [
        column
        for column in context.columns
        if column != "district"
    ]


    for column in numeric_columns:

        context[column] = pd.to_numeric(
            context[column],
            errors="coerce",
        )


    # ========================================================
    # Quality checks
    # ========================================================

    print()
    print("=" * 70)

    print(
        "HISTORICAL CONTEXT QUALITY CHECK"
    )

    print("=" * 70)


    print()
    print(
        f"Final districts: "
        f"{len(context)}"
    )


    print()
    print(
        "Districts:"
    )

    print(
        context[
            "district"
        ]
        .to_string(
            index=False
        )
    )


    print()
    print(
        "Missing values:"
    )


    missing = (
        context
        .isna()
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


    print()
    print(
        "Historical context:"
    )


    print(
        context.to_string(
            index=False
        )
    )


    # ========================================================
    # Check district coverage
    # ========================================================

    actual_districts = set(
        context["district"]
    )


    expected_districts = set(
        TARGET_DISTRICTS
    )


    missing_districts = (
        expected_districts
        - actual_districts
    )


    if missing_districts:

        raise ValueError(
            "Missing target districts: "
            + str(
                sorted(
                    missing_districts
                )
            )
        )


    # ========================================================
    # Save
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    context.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    print()
    print(
        "Saved historical context:"
    )


    print(
        OUTPUT_FILE
    )


    print()
    print("=" * 70)

    print(
        "Historical context completed."
    )

    print("=" * 70)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()