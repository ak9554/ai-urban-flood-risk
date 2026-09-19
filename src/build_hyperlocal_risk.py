from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPATIAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_spatial_susceptibility_with_coordinates.csv"
)

RISK_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_hgb_v4_predictions_2022_2023.csv"
)

HISTORICAL_CONTEXT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_historical_context.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_hyperlocal_risk.csv"
)


# ============================================================
# Default scenario date
# ============================================================

DEFAULT_SCENARIO_DATE = "2023-07-19"


# ============================================================
# Prototype composite weights
# ============================================================

WEIGHT_TEMPORAL_RISK = 0.45
WEIGHT_SPATIAL_SUSCEPTIBILITY = 0.40
WEIGHT_VULNERABILITY = 0.15


# ============================================================
# Helper
# ============================================================

def normalize_0_100(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            50.0,
            index=series.index,
        )

    return (
        (
            series - minimum
        )
        /
        (
            maximum - minimum
        )
        * 100
    )


# ============================================================
# Get scenario date
# ============================================================

def get_scenario_date():

    # --------------------------------------------------------
    # If a date was supplied from the command line:
    #
    # python src\build_hyperlocal_risk.py 2023-07-19
    # --------------------------------------------------------

    if len(sys.argv) > 1:

        scenario_date = sys.argv[1]

    else:

        scenario_date = DEFAULT_SCENARIO_DATE


    try:

        parsed_date = pd.Timestamp(
            scenario_date
        )

    except Exception:

        raise ValueError(
            "Invalid scenario date. "
            "Use YYYY-MM-DD, for example 2023-07-19."
        )


    return parsed_date


# ============================================================
# Load spatial susceptibility
# ============================================================

def load_spatial_data():

    spatial = pd.read_csv(
        SPATIAL_FILE
    )

    print(
        f"Spatial grid rows: "
        f"{len(spatial):,}"
    )

    return spatial


# ============================================================
# Load HGB V4 predictions
# ============================================================

def load_risk_predictions():

    predictions = pd.read_csv(
        RISK_PREDICTIONS_FILE
    )

    print(
        f"Prediction rows: "
        f"{len(predictions):,}"
    )


    required_columns = [

        "district",
        "date",
        "temporal_risk_score",

        "rainfall_24h",
        "rainfall_3day",
        "rainfall_7day",

        "flood_event",
    ]


    missing_columns = [

        column
        for column in required_columns
        if column not in predictions.columns
    ]


    if missing_columns:

        raise ValueError(
            "Prediction file is missing columns: "
            + str(missing_columns)
        )


    predictions["date"] = pd.to_datetime(
        predictions["date"]
    )


    predictions["district"] = (
        predictions["district"]
        .astype(str)
        .str.upper()
        .str.strip()
    )


    numeric_columns = [

        "temporal_risk_score",
        "rainfall_24h",
        "rainfall_3day",
        "rainfall_7day",
        "flood_event",
    ]


    for column in numeric_columns:

        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="coerce",
        )


    # --------------------------------------------------------
    # HGB score should be between 0 and 1.
    # --------------------------------------------------------

    invalid_scores = predictions[
        (
            predictions[
                "temporal_risk_score"
            ] < 0
        )
        |
        (
            predictions[
                "temporal_risk_score"
            ] > 1
        )
    ]


    if len(invalid_scores) > 0:

        raise ValueError(
            "Invalid temporal risk scores found. "
            "Expected values between 0 and 1."
        )


    return predictions


# ============================================================
# Load historical context
# ============================================================

def load_historical_context():

    if not HISTORICAL_CONTEXT_FILE.exists():

        raise FileNotFoundError(
            "Historical context file was not found:\n"
            f"{HISTORICAL_CONTEXT_FILE}"
        )


    context = pd.read_csv(
        HISTORICAL_CONTEXT_FILE
    )


    print(
        f"Historical context rows: "
        f"{len(context):,}"
    )


    context["district"] = (
        context["district"]
        .astype(str)
        .str.upper()
        .str.strip()
    )


    return context


# ============================================================
# Build vulnerability score
# ============================================================

def build_vulnerability_score(
    spatial,
    context,
):

    result = spatial.copy()


    result["district"] = (
        result["district"]
        .astype(str)
        .str.upper()
        .str.strip()
    )


    context = context.copy()


    context["district"] = (
        context["district"]
        .astype(str)
        .str.upper()
        .str.strip()
    )


    # --------------------------------------------------------
    # Historical flooded area
    # --------------------------------------------------------

    area_score = normalize_0_100(
        context[
            "corrected_flooded_area_percent"
        ]
    )


    # --------------------------------------------------------
    # Historical duration
    # --------------------------------------------------------

    duration_score = normalize_0_100(
        context[
            "historical_mean_flood_duration"
        ]
    )


    # --------------------------------------------------------
    # Population
    # --------------------------------------------------------

    population_score = normalize_0_100(
        context[
            "population"
        ]
    )


    # --------------------------------------------------------
    # Historical flood frequency
    # --------------------------------------------------------

    frequency_score = normalize_0_100(
        context[
            "historical_flood_days_2015_2023"
        ]
    )


    # --------------------------------------------------------
    # Prototype vulnerability index
    # --------------------------------------------------------

    context[
        "vulnerability_score"
    ] = (

        area_score
        + duration_score
        + population_score
        + frequency_score

    ) / 4.0


    # --------------------------------------------------------
    # Merge onto grid
    # --------------------------------------------------------

    result = result.merge(

        context[
            [
                "district",
                "historical_flooded_area_percent",
                "historical_mean_flood_duration",
                "population",
                "historical_flood_days_2015_2023",
                "vulnerability_score",
            ]
        ],

        on="district",

        how="left",

        validate="many_to_one",
    )


    return result


# ============================================================
# Select scenario
# ============================================================

def select_scenario(
    predictions,
    scenario_date,
):

    available_dates = (
        predictions[
            "date"
        ]
        .drop_duplicates()
        .sort_values()
    )


    # --------------------------------------------------------
    # Validate requested date
    # --------------------------------------------------------

    if scenario_date not in set(
        available_dates
    ):

        earliest = available_dates.min()
        latest = available_dates.max()

        raise ValueError(

            f"Scenario date "
            f"{scenario_date.date()} "
            "is not available.\n\n"

            f"Available date range: "
            f"{earliest.date()} "
            f"to "
            f"{latest.date()}"
        )


    scenario = predictions[
        predictions["date"]
        == scenario_date
    ].copy()


    print()

    print(
        f"Scenario date: "
        f"{scenario_date.date()}"
    )


    print(
        f"District predictions: "
        f"{len(scenario)}"
    )


    # --------------------------------------------------------
    # Scenario rainfall and model signal
    # --------------------------------------------------------

    print()

    print(
        "Scenario rainfall and HGB V4 model signal:"
    )


    display_columns = [

        "district",

        "rainfall_24h",
        "rainfall_3day",
        "rainfall_7day",

        "temporal_risk_score",

        "flood_event",
    ]


    print(

        scenario[
            display_columns
        ]
        .sort_values(
            "temporal_risk_score",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )


    return scenario


# ============================================================
# Build hyperlocal risk
# ============================================================

def calculate_hyperlocal_risk(
    spatial,
    predictions,
    context,
    scenario_date,
):

    print()

    print(
        "Building vulnerability context..."
    )


    spatial = build_vulnerability_score(
        spatial,
        context,
    )


    scenario = select_scenario(
        predictions,
        scenario_date,
    )


    # --------------------------------------------------------
    # Merge district temporal signal onto 1-km grid
    # --------------------------------------------------------

    result = spatial.merge(

        scenario[
            [
                "district",
                "date",
                "rainfall_24h",
                "rainfall_3day",
                "rainfall_7day",
                "temporal_risk_score",
                "flood_event",
            ]
        ],

        on="district",

        how="left",

        validate="many_to_one",
    )


    # --------------------------------------------------------
    # Convert model score from 0-1 to a 0-100 index.
    #
    # This is NOT a calibrated probability.
    # --------------------------------------------------------

    result[
        "temporal_risk_index"
    ] = (

        result[
            "temporal_risk_score"
        ]

        .clip(
            lower=0,
            upper=1,
        )

        * 100
    )


    # --------------------------------------------------------
    # Hyperlocal composite score
    # --------------------------------------------------------

    result[
        "hyperlocal_risk_score"
    ] = (

        WEIGHT_TEMPORAL_RISK
        * result[
            "temporal_risk_index"
        ]

        +

        WEIGHT_SPATIAL_SUSCEPTIBILITY
        * result[
            "susceptibility_score"
        ]

        +

        WEIGHT_VULNERABILITY
        * result[
            "vulnerability_score"
        ]
    )


    # --------------------------------------------------------
    # Risk category
    # --------------------------------------------------------

    result[
        "hyperlocal_risk_category"
    ] = pd.cut(

        result[
            "hyperlocal_risk_score"
        ],

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


    # --------------------------------------------------------
    # Action priority
    # --------------------------------------------------------

    result[
        "action_priority"
    ] = pd.cut(

        result[
            "hyperlocal_risk_score"
        ],

        bins=[
            -np.inf,
            30,
            50,
            70,
            np.inf,
        ],

        labels=[
            "Routine Monitoring",
            "Monitor",
            "High Priority",
            "Immediate Review",
        ],
    )


    # ========================================================
    # Explanation
    # ========================================================

    def create_explanation(row):

        explanations = []


        # ----------------------------------------------------
        # Spatial driver
        # ----------------------------------------------------

        if pd.notna(
            row.get(
                "dominant_driver"
            )
        ):

            explanations.append(
                row[
                    "dominant_driver"
                ]
            )


        # ----------------------------------------------------
        # Rainfall
        # ----------------------------------------------------

        if row[
            "rainfall_24h"
        ] >= 20:

            explanations.append(
                "High 24-hour rainfall"
            )

        elif row[
            "rainfall_24h"
        ] >= 10:

            explanations.append(
                "Elevated 24-hour rainfall"
            )


        # ----------------------------------------------------
        # Temporal model signal
        # ----------------------------------------------------

        if row[
            "temporal_risk_index"
        ] >= 20:

            explanations.append(
                "Elevated temporal model signal"
            )

        elif row[
            "temporal_risk_index"
        ] >= 5:

            explanations.append(
                "Moderate temporal model signal"
            )


        # ----------------------------------------------------
        # Historical vulnerability
        # ----------------------------------------------------

        if row[
            "vulnerability_score"
        ] >= 70:

            explanations.append(
                "High historical vulnerability context"
            )

        elif row[
            "vulnerability_score"
        ] >= 40:

            explanations.append(
                "Moderate historical vulnerability context"
            )


        # ----------------------------------------------------
        # Default
        # ----------------------------------------------------

        if len(
            explanations
        ) == 0:

            return (
                "No major prototype risk driver "
                "exceeded the selected thresholds."
            )


        return "; ".join(
            explanations
        )


    result[
        "risk_explanation"
    ] = result.apply(
        create_explanation,
        axis=1,
    )


    return result


# ============================================================
# Quality checks
# ============================================================

def quality_checks(
    result,
    scenario_date,
):

    print()

    print("=" * 70)

    print(
        "HYPERLOCAL RISK QUALITY CHECK"
    )

    print("=" * 70)


    print()

    print(
        f"Scenario date: "
        f"{scenario_date.date()}"
    )


    print()

    print(
        f"Grid cells: "
        f"{len(result):,}"
    )


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    important_columns = [

        "grid_id",
        "district",
        "latitude",
        "longitude",

        "susceptibility_score",
        "vulnerability_score",

        "temporal_risk_index",

        "hyperlocal_risk_score",
    ]


    missing = (
        result[
            important_columns
        ]
        .isna()
        .sum()
    )


    missing = missing[
        missing > 0
    ]


    print()

    print(
        "Missing values:"
    )


    if len(
        missing
    ) == 0:

        print(
            "No missing values."
        )

    else:

        print(
            missing.to_string()
        )


    # --------------------------------------------------------
    # Temporal model
    # --------------------------------------------------------

    print()

    print(
        "Temporal model score statistics (0-100 index):"
    )


    print(

        result[
            "temporal_risk_index"
        ]
        .describe()
        .round(3)
        .to_string()
    )


    # --------------------------------------------------------
    # Spatial susceptibility
    # --------------------------------------------------------

    print()

    print(
        "Spatial susceptibility statistics:"
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
    # Vulnerability
    # --------------------------------------------------------

    print()

    print(
        "Vulnerability statistics:"
    )


    print(

        result[
            "vulnerability_score"
        ]
        .describe()
        .round(3)
        .to_string()
    )


    # --------------------------------------------------------
    # Hyperlocal score
    # --------------------------------------------------------

    print()

    print(
        "Hyperlocal risk statistics:"
    )


    print(

        result[
            "hyperlocal_risk_score"
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
        "Hyperlocal risk categories:"
    )


    print(

        result[
            "hyperlocal_risk_category"
        ]
        .value_counts(
            sort=False
        )
        .to_string()
    )


    # --------------------------------------------------------
    # Action priority
    # --------------------------------------------------------

    print()

    print(
        "Action priority:"
    )


    print(

        result[
            "action_priority"
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
        "District hyperlocal risk summary:"
    )


    district_summary = (

        result

        .groupby(
            "district",
            observed=True,
        )

        [
            "hyperlocal_risk_score"
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
        district_summary.to_string()
    )


    # --------------------------------------------------------
    # Top cells
    # --------------------------------------------------------

    print()

    print(
        "Top 10 highest-risk grid cells:"
    )


    top_cells = (

        result

        .sort_values(
            "hyperlocal_risk_score",
            ascending=False,
        )

        [

            [
                "grid_id",
                "district",
                "hyperlocal_risk_score",
                "susceptibility_score",
                "temporal_risk_index",
                "vulnerability_score",
                "dominant_driver",
                "risk_explanation",
            ]

        ]

        .head(10)
    )


    print(
        top_cells.to_string(
            index=False
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "FloodSense AI - HGB V4 Hyperlocal Risk Scenario"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Determine scenario date
    # --------------------------------------------------------

    scenario_date = get_scenario_date()


    # --------------------------------------------------------
    # Load spatial
    # --------------------------------------------------------

    print()

    print(
        "Loading spatial susceptibility..."
    )


    spatial = load_spatial_data()


    # --------------------------------------------------------
    # Load temporal predictions
    # --------------------------------------------------------

    print()

    print(
        "Loading HGB V4 temporal predictions..."
    )


    predictions = load_risk_predictions()


    # --------------------------------------------------------
    # Load historical context
    # --------------------------------------------------------

    print()

    print(
        "Loading historical context..."
    )


    context = load_historical_context()


    # --------------------------------------------------------
    # Build hyperlocal risk
    # --------------------------------------------------------

    result = calculate_hyperlocal_risk(

        spatial,

        predictions,

        context,

        scenario_date,
    )


    # --------------------------------------------------------
    # Quality checks
    # --------------------------------------------------------

    quality_checks(

        result,

        scenario_date,
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
        "Saved HGB V4 hyperlocal scenario:"
    )


    print(
        OUTPUT_FILE
    )


    print()

    print("=" * 70)

    print(
        "HGB V4 hyperlocal scenario completed."
    )

    print("=" * 70)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()