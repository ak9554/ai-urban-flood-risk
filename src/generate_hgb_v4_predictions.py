from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


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
    / "ncr_hgb_v4_predictions_2022_2023.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "ncr_hgb_v4_temporal_model.joblib"
)


# ============================================================
# Configuration
# ============================================================

TRAIN_END_DATE = "2021-12-31"

TEST_START_DATE = "2022-01-01"

# This threshold was obtained during the earlier V4 experiment.
# It is used only for the alert flag.
ALERT_THRESHOLD = 0.040


# ============================================================
# Load data
# ============================================================

def load_data():

    df = pd.read_csv(
        INPUT_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["district"] = (
        df["district"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df = df.sort_values(
        [
            "district",
            "date",
        ]
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# Time-safe feature engineering
# ============================================================

def build_features(df):

    result = df.copy()

    # --------------------------------------------------------
    # Basic calendar features
    # --------------------------------------------------------

    result["month"] = (
        result["date"].dt.month
    )

    result["day_of_year"] = (
        result["date"].dt.dayofyear
    )

    result["month_sin"] = np.sin(
        2
        * np.pi
        * result["month"]
        / 12
    )

    result["month_cos"] = np.cos(
        2
        * np.pi
        * result["month"]
        / 12
    )

    result["day_of_year_sin"] = np.sin(
        2
        * np.pi
        * result["day_of_year"]
        / 365.25
    )

    result["day_of_year_cos"] = np.cos(
        2
        * np.pi
        * result["day_of_year"]
        / 365.25
    )

    result["monsoon_flag"] = (
        result["month"].isin(
            [6, 7, 8, 9]
        )
        .astype(int)
    )


    # --------------------------------------------------------
    # Rainfall accumulation relationships
    # --------------------------------------------------------

    result[
        "rainfall_24h_ratio_3day"
    ] = np.where(
        result["rainfall_3day"] > 0,
        result["rainfall_24h"]
        / result["rainfall_3day"],
        0,
    )


    result[
        "rainfall_24h_ratio_7day"
    ] = np.where(
        result["rainfall_7day"] > 0,
        result["rainfall_24h"]
        / result["rainfall_7day"],
        0,
    )


    result[
        "rainfall_3day_ratio_7day"
    ] = np.where(
        result["rainfall_7day"] > 0,
        result["rainfall_3day"]
        / result["rainfall_7day"],
        0,
    )


    result[
        "rainfall_3day_daily_average"
    ] = (
        result["rainfall_3day"]
        / 3
    )


    result[
        "rainfall_7day_daily_average"
    ] = (
        result["rainfall_7day"]
        / 7
    )


    result[
        "rainfall_recent_excess"
    ] = (
        result["rainfall_24h"]
        -
        (
            result["rainfall_7day"]
            / 7
        )
    )


    # --------------------------------------------------------
    # Previous rainfall
    #
    # shift(1) means yesterday for each district.
    # This prevents today's rainfall from leaking backward.
    # --------------------------------------------------------

    grouped = result.groupby(
        "district",
        group_keys=False,
    )


    result[
        "rainfall_previous_24h"
    ] = grouped[
        "rainfall_24h"
    ].shift(1)


    result[
        "rainfall_previous_3day"
    ] = grouped[
        "rainfall_3day"
    ].shift(1)


    result[
        "rainfall_previous_7day"
    ] = grouped[
        "rainfall_7day"
    ].shift(1)


    # --------------------------------------------------------
    # Rainfall change
    # --------------------------------------------------------

    result[
        "rainfall_24h_change"
    ] = (
        result["rainfall_24h"]
        -
        result[
            "rainfall_previous_24h"
        ]
    )


    result[
        "rainfall_3day_change"
    ] = (
        result["rainfall_3day"]
        -
        result[
            "rainfall_previous_3day"
        ]
    )


    result[
        "rainfall_7day_change"
    ] = (
        result["rainfall_7day"]
        -
        result[
            "rainfall_previous_7day"
        ]
    )


    # --------------------------------------------------------
    # Time-safe rainfall percentile
    #
    # For each row, only rainfall observations BEFORE that
    # date are used to calculate the percentile.
    # --------------------------------------------------------

    def expanding_percentile(series):

        values = (
            series
            .shift(1)
            .expanding()
            .rank(
                pct=True
            )
        )

        return values


    result[
        "rainfall_24h_percentile"
    ] = grouped[
        "rainfall_24h"
    ].transform(
        expanding_percentile
    )


    result[
        "rainfall_3day_percentile"
    ] = grouped[
        "rainfall_3day"
    ].transform(
        expanding_percentile
    )


    result[
        "rainfall_7day_percentile"
    ] = grouped[
        "rainfall_7day"
    ].transform(
        expanding_percentile
    )


    # --------------------------------------------------------
    # Historical flood-event memory
    #
    # shift(1) ensures today's flood label is never used to
    # predict today's flood.
    # --------------------------------------------------------

    result[
        "previous_flood_event"
    ] = grouped[
        "flood_event"
    ].shift(1)


    # --------------------------------------------------------
    # Historical flood frequency
    #
    # All counts are based only on previous observations.
    # --------------------------------------------------------

    result[
        "historical_flood_count"
    ] = (
        grouped[
            "flood_event"
        ]
        .transform(
            lambda s:
            s.shift(1)
            .expanding()
            .sum()
        )
    )


    result[
        "previous_365d_flood_count"
    ] = (
        result
        .groupby("district")[
            "flood_event"
        ]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                window=365,
                min_periods=1,
            )
            .sum()
        )
    )


    result[
        "previous_3yr_flood_count"
    ] = (
        result
        .groupby("district")[
            "flood_event"
        ]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                window=1095,
                min_periods=1,
            )
            .sum()
        )
    )


    # --------------------------------------------------------
    # Fill values created by lagged features.
    #
    # Missing lagged values occur only at the beginning of
    # each district's historical record.
    # --------------------------------------------------------

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns


    result[
        numeric_columns
    ] = result[
        numeric_columns
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


    result[
        numeric_columns
    ] = result[
        numeric_columns
    ].fillna(0)


    return result


# ============================================================
# Prepare model matrix
# ============================================================

def prepare_model_data(
    df
):

    feature_columns = [

        # Rainfall
        "rainfall_24h",
        "rainfall_3day",
        "rainfall_7day",

        # Rainfall relationships
        "rainfall_24h_ratio_3day",
        "rainfall_24h_ratio_7day",
        "rainfall_3day_ratio_7day",
        "rainfall_3day_daily_average",
        "rainfall_7day_daily_average",
        "rainfall_recent_excess",

        # Rainfall history
        "rainfall_previous_24h",
        "rainfall_previous_3day",
        "rainfall_previous_7day",

        "rainfall_24h_change",
        "rainfall_3day_change",
        "rainfall_7day_change",

        "rainfall_24h_percentile",
        "rainfall_3day_percentile",
        "rainfall_7day_percentile",

        # Calendar
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "monsoon_flag",

        # Flood history
        "previous_flood_event",
        "historical_flood_count",
        "previous_365d_flood_count",
        "previous_3yr_flood_count",
    ]


    # --------------------------------------------------------
    # District one-hot encoding
    # --------------------------------------------------------

    district_dummies = pd.get_dummies(
        df["district"],
        prefix="district",
        dtype=int,
    )


    X = pd.concat(
        [
            df[
                feature_columns
            ],
            district_dummies,
        ],
        axis=1,
    )


    y = df[
        "flood_event"
    ].astype(int)


    return (
        X,
        y,
        list(X.columns),
    )


# ============================================================
# Train model
# ============================================================

def train_model(
    X_train,
    y_train,
):

    model = HistGradientBoostingClassifier(

        learning_rate=0.05,

        max_iter=300,

        max_leaf_nodes=15,

        min_samples_leaf=20,

        l2_regularization=1.0,

        random_state=42,
    )


    model.fit(
        X_train,
        y_train,
    )


    return model


# ============================================================
# Evaluate
# ============================================================

def evaluate(
    model,
    X_test,
    y_test,
):

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    predictions = (
        probabilities
        >= ALERT_THRESHOLD
    ).astype(int)


    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )


    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )


    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )


    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )


    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )


    cm = confusion_matrix(
        y_test,
        predictions,
    )


    print()
    print("=" * 70)

    print(
        "HGB V4 TEST PERFORMANCE"
    )

    print("=" * 70)


    print(
        f"Alert threshold: "
        f"{ALERT_THRESHOLD:.3f}"
    )


    print(
        f"Precision: "
        f"{precision:.4f}"
    )


    print(
        f"Recall: "
        f"{recall:.4f}"
    )


    print(
        f"F1: "
        f"{f1:.4f}"
    )


    print(
        f"ROC-AUC: "
        f"{roc_auc:.4f}"
    )


    print(
        f"PR-AUC: "
        f"{pr_auc:.4f}"
    )


    print()
    print(
        "Confusion matrix:"
    )


    print(
        cm
    )


    print()
    print(
        f"Alerts: "
        f"{predictions.sum():,}"
    )


    print(
        f"Actual flood events: "
        f"{y_test.sum():,}"
    )


    return probabilities


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "FloodSense AI - HGB V4 Production Model"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print()
    print(
        "Loading district-day training data..."
    )


    df = load_data()


    print(
        f"Rows: "
        f"{len(df):,}"
    )


    print(
        f"Flood events: "
        f"{df['flood_event'].sum():,}"
    )


    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print()
    print(
        "Building time-safe features..."
    )


    df_features = build_features(
        df
    )


    # --------------------------------------------------------
    # Split by time
    # --------------------------------------------------------

    train_mask = (
        df_features["date"]
        <= pd.Timestamp(
            TRAIN_END_DATE
        )
    )


    test_mask = (
        df_features["date"]
        >= pd.Timestamp(
            TEST_START_DATE
        )
    )


    train_df = df_features[
        train_mask
    ].copy()


    test_df = df_features[
        test_mask
    ].copy()


    print()
    print(
        "Temporal split:"
    )


    print(
        f"Training: "
        f"{train_df['date'].min().date()}"
        f" to "
        f"{train_df['date'].max().date()}"
    )


    print(
        f"Testing: "
        f"{test_df['date'].min().date()}"
        f" to "
        f"{test_df['date'].max().date()}"
    )


    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )


    print(
        f"Training flood events: "
        f"{train_df['flood_event'].sum():,}"
    )


    print(
        f"Testing rows: "
        f"{len(test_df):,}"
    )


    print(
        f"Testing flood events: "
        f"{test_df['flood_event'].sum():,}"
    )


    # --------------------------------------------------------
    # Model matrices
    # --------------------------------------------------------

    X_train, y_train, feature_columns = (
        prepare_model_data(
            train_df
        )
    )


    X_test, y_test, _ = (
        prepare_model_data(
            test_df
        )
    )


    # --------------------------------------------------------
    # Align columns
    # --------------------------------------------------------

    X_test = X_test.reindex(
        columns=feature_columns,
        fill_value=0,
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print(
        "Training HistGradientBoosting model..."
    )


    model = train_model(
        X_train,
        y_train,
    )


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    test_probabilities = evaluate(
        model,
        X_test,
        y_test,
    )


    # --------------------------------------------------------
    # Create prediction output
    # --------------------------------------------------------

    predictions = test_df[
        [
            "district",
            "date",
            "rainfall_24h",
            "rainfall_3day",
            "rainfall_7day",
            "flood_event",
        ]
    ].copy()


    predictions[
        "temporal_risk_score"
    ] = test_probabilities


    predictions[
        "temporal_risk_score_percent"
    ] = (
        test_probabilities
        * 100
    )


    predictions[
        "flood_alert"
    ] = (
        test_probabilities
        >= ALERT_THRESHOLD
    ).astype(int)


    predictions[
        "risk_level"
    ] = pd.cut(

        test_probabilities,

        bins=[
            -np.inf,
            0.01,
            0.04,
            0.10,
            0.25,
            np.inf,
        ],

        labels=[
            "LOW",
            "MODERATE",
            "ELEVATED",
            "HIGH",
            "VERY HIGH",
        ],
    )


    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    predictions.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    import joblib


    model_package = {

        "model": model,

        "feature_columns":
            feature_columns,

        "alert_threshold":
            ALERT_THRESHOLD,

        "train_end_date":
            TRAIN_END_DATE,

        "test_start_date":
            TEST_START_DATE,

        "model_type":
            "HistGradientBoostingClassifier",

        "note":
            (
                "Prototype district-day temporal "
                "flood-risk model. Risk score is "
                "not calibrated emergency probability."
            ),
    }


    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    joblib.dump(
        model_package,
        MODEL_FILE,
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "OUTPUT SUMMARY"
    )

    print("=" * 70)


    print()
    print(
        f"Prediction rows: "
        f"{len(predictions):,}"
    )


    print(
        f"Prediction date range: "
        f"{predictions['date'].min().date()}"
        f" to "
        f"{predictions['date'].max().date()}"
    )


    print()
    print(
        "Risk-score statistics:"
    )


    print(
        predictions[
            "temporal_risk_score"
        ]
        .describe()
        .round(6)
        .to_string()
    )


    print()
    print(
        "Flood alerts:"
    )


    print(
        predictions[
            "flood_alert"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


    print()
    print(
        "Saved predictions:"
    )


    print(
        OUTPUT_FILE
    )


    print()
    print(
        "Saved model:"
    )


    print(
        MODEL_FILE
    )


    print()
    print("=" * 70)

    print(
        "HGB V4 production model completed."
    )

    print("=" * 70)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()