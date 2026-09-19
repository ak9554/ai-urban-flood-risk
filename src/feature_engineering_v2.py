from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_district_day_training.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_district_day_features_v2.csv"
)


# ============================================================
# Configuration
# ============================================================

TARGET_COLUMN = "flood_event"
DATE_COLUMN = "date"
DISTRICT_COLUMN = "district"


# ============================================================
# Load data
# ============================================================

def load_data():

    data = pd.read_csv(
        INPUT_FILE,
        parse_dates=[DATE_COLUMN],
    )

    data = (
        data
        .sort_values(
            [DISTRICT_COLUMN, DATE_COLUMN]
        )
        .reset_index(drop=True)
    )

    return data


# ============================================================
# Create rainfall features
# ============================================================

def create_rainfall_features(data):

    result = data.copy()


    # --------------------------------------------------------
    # Basic rainfall features
    # --------------------------------------------------------

    rainfall_24h = result["rainfall_24h"]

    rainfall_3day = result["rainfall_3day"]

    rainfall_7day = result["rainfall_7day"]


    # --------------------------------------------------------
    # Rainfall concentration
    # --------------------------------------------------------

    result["rainfall_24h_ratio_3day"] = (
        rainfall_24h
        / rainfall_3day.replace(0, np.nan)
    )

    result["rainfall_24h_ratio_7day"] = (
        rainfall_24h
        / rainfall_7day.replace(0, np.nan)
    )

    result["rainfall_3day_ratio_7day"] = (
        rainfall_3day
        / rainfall_7day.replace(0, np.nan)
    )


    # --------------------------------------------------------
    # Daily contribution to accumulated rainfall
    # --------------------------------------------------------

    result["rainfall_3day_daily_average"] = (
        rainfall_3day / 3.0
    )

    result["rainfall_7day_daily_average"] = (
        rainfall_7day / 7.0
    )


    # --------------------------------------------------------
    # Recent rainfall compared with accumulated rainfall
    # --------------------------------------------------------

    result["rainfall_recent_excess"] = (
        rainfall_24h
        - (
            rainfall_7day
            / 7.0
        )
    )


    # --------------------------------------------------------
    # Previous-day rainfall
    #
    # shift(1) means:
    # "what was the rainfall yesterday?"
    #
    # We calculate this separately for every district.
    # --------------------------------------------------------

    result["rainfall_previous_24h"] = (
        result
        .groupby(DISTRICT_COLUMN)["rainfall_24h"]
        .shift(1)
    )


    # --------------------------------------------------------
    # Previous 3-day and 7-day accumulated rainfall
    # --------------------------------------------------------

    result["rainfall_previous_3day"] = (
        result
        .groupby(DISTRICT_COLUMN)["rainfall_3day"]
        .shift(1)
    )

    result["rainfall_previous_7day"] = (
        result
        .groupby(DISTRICT_COLUMN)["rainfall_7day"]
        .shift(1)
    )


    # --------------------------------------------------------
    # Change in rainfall
    # --------------------------------------------------------

    result["rainfall_24h_change"] = (
        rainfall_24h
        - result["rainfall_previous_24h"]
    )

    result["rainfall_3day_change"] = (
        rainfall_3day
        - result["rainfall_previous_3day"]
    )

    result["rainfall_7day_change"] = (
        rainfall_7day
        - result["rainfall_previous_7day"]
    )


    # --------------------------------------------------------
    # Percentage change
    # --------------------------------------------------------

    result["rainfall_24h_percent_change"] = (
        (
            rainfall_24h
            - result["rainfall_previous_24h"]
        )
        / result["rainfall_previous_24h"].replace(
            0,
            np.nan,
        )
    )


    # ========================================================
    # Historical rainfall percentiles
    #
    # IMPORTANT:
    # These are calculated separately for each district.
    #
    # We use the historical rainfall distribution from the
    # complete 2015-2023 dataset as a descriptive feature.
    # We will later test whether this should be replaced by
    # strictly past-only percentiles for deployment.
    # ========================================================

    district_groups = result.groupby(
        DISTRICT_COLUMN
    )


    result["rainfall_24h_percentile"] = (
        district_groups["rainfall_24h"]
        .rank(
            pct=True,
            method="average",
        )
    )


    result["rainfall_3day_percentile"] = (
        district_groups["rainfall_3day"]
        .rank(
            pct=True,
            method="average",
        )
    )


    result["rainfall_7day_percentile"] = (
        district_groups["rainfall_7day"]
        .rank(
            pct=True,
            method="average",
        )
    )


    # --------------------------------------------------------
    # Extreme rainfall flags
    #
    # These are useful because flood events may be associated
    # with unusually high rainfall rather than just high
    # absolute rainfall.
    # --------------------------------------------------------

    result["rainfall_24h_extreme"] = (
        result["rainfall_24h_percentile"]
        >= 0.90
    ).astype(int)


    result["rainfall_3day_extreme"] = (
        result["rainfall_3day_percentile"]
        >= 0.90
    ).astype(int)


    result["rainfall_7day_extreme"] = (
        result["rainfall_7day_percentile"]
        >= 0.90
    ).astype(int)


    result["rainfall_24h_very_extreme"] = (
        result["rainfall_24h_percentile"]
        >= 0.95
    ).astype(int)


    result["rainfall_24h_extremely_extreme"] = (
        result["rainfall_24h_percentile"]
        >= 0.99
    ).astype(int)


    # ========================================================
    # Seasonal features
    # ========================================================

    dates = pd.to_datetime(
        result[DATE_COLUMN]
    )

    month = dates.dt.month

    day_of_year = dates.dt.dayofyear


    result["month"] = month


    result["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )


    result["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )


    result["day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )


    result["day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )


    # --------------------------------------------------------
    # Monsoon indicator
    #
    # June-September = 1
    # Other months = 0
    # --------------------------------------------------------

    result["monsoon_flag"] = (
        month.isin(
            [6, 7, 8, 9]
        )
    ).astype(int)


    # ========================================================
    # Clean numerical values
    # ========================================================

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


    # Missing previous-day values occur naturally at the
    # beginning of each district's time series.
    #
    # We keep them as 0 for this feature table.
    #

    result = result.fillna(0)


    return result


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)

    print(
        "FloodSense AI - Rainfall Feature Engineering V2"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_data()


    print()
    print(
        f"Input rows: {len(data):,}"
    )


    print(
        f"Input columns: {len(data.columns)}"
    )


    # --------------------------------------------------------
    # Create features
    # --------------------------------------------------------

    result = create_rainfall_features(
        data
    )


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print()
    print(
        f"Output rows: {len(result):,}"
    )


    print(
        f"Output columns: {len(result.columns)}"
    )


    print()
    print("New features:")


    original_columns = set(
        data.columns
    )


    new_columns = [
        column
        for column in result.columns
        if column not in original_columns
    ]


    for column in new_columns:

        print(
            f"  - {column}"
        )


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print()
    print("Missing values:")


    missing = (
        result.isna()
        .sum()
    )


    print(
        missing[
            missing > 0
        ].to_string()
    )


    if missing.sum() == 0:

        print(
            "No missing values."
        )


    # --------------------------------------------------------
    # Flood label check
    # --------------------------------------------------------

    print()
    print("Flood label distribution:")


    print(
        result[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .to_string()
    )


    # --------------------------------------------------------
    # Rainfall feature statistics
    # --------------------------------------------------------

    print()
    print("Rainfall feature summary:")


    summary_columns = [
        "rainfall_24h",
        "rainfall_3day",
        "rainfall_7day",
        "rainfall_24h_ratio_7day",
        "rainfall_3day_ratio_7day",
        "rainfall_24h_percentile",
        "rainfall_3day_percentile",
        "rainfall_7day_percentile",
    ]


    print(
        result[
            summary_columns
        ].describe()
        .round(3)
        .to_string()
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
        f"Saved feature dataset:"
    )

    print(
        OUTPUT_FILE
    )


    print()
    print("=" * 60)

    print(
        "Rainfall feature engineering completed."
    )

    print("=" * 60)


if __name__ == "__main__":

    main()