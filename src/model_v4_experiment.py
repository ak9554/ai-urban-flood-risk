from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_district_day_features_v2.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_v4_comparison.csv"
)


# ============================================================
# Configuration
# ============================================================

TARGET = "flood_event"

TRAIN_END = pd.Timestamp(
    "2021-12-31"
)

TEST_START = pd.Timestamp(
    "2022-01-01"
)


# ============================================================
# Load data
# ============================================================

def load_data():

    data = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    data = (
        data
        .sort_values(
            ["district", "date"]
        )
        .reset_index(drop=True)
    )

    return data


# ============================================================
# Create time-safe rainfall percentile features
# ============================================================

def create_time_safe_rainfall_features(data):

    df = data.copy()

    percentile_columns = [
        "rainfall_24h",
        "rainfall_3day",
        "rainfall_7day",
    ]

    for column in percentile_columns:

        safe_percentile_name = (
            column
            + "_past_percentile"
        )

        values = []

        for district, group in df.groupby(
            "district",
            sort=False,
        ):

            group = group.sort_values(
                "date"
            ).copy()

            rainfall_values = (
                group[column]
                .astype(float)
                .to_numpy()
            )

            past_percentiles = []

            for i in range(
                len(rainfall_values)
            ):

                current_value = (
                    rainfall_values[i]
                )

                # No previous observations
                if i == 0:

                    past_percentiles.append(
                        0.5
                    )

                    continue

                previous_values = (
                    rainfall_values[:i]
                )

                percentile = (
                    np.mean(
                        previous_values
                        <= current_value
                    )
                )

                past_percentiles.append(
                    percentile
                )

            group[
                safe_percentile_name
            ] = past_percentiles

            values.append(
                group[
                    [
                        "district",
                        "date",
                        safe_percentile_name,
                    ]
                ]
            )

        percentile_table = pd.concat(
            values,
            ignore_index=True,
        )

        df = df.merge(
            percentile_table,
            on=[
                "district",
                "date",
            ],
            how="left",
        )

    return df


# ============================================================
# Create time-safe historical flood features
# ============================================================

def create_flood_history_features(data):

    df = data.copy()

    df = df.sort_values(
        [
            "district",
            "date",
        ]
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Previous flood event
    #
    # shift(1) means today's flood label is NOT included.
    # --------------------------------------------------------

    df["previous_flood_event"] = (
        df.groupby(
            "district"
        )["flood_event"]
        .shift(1)
        .fillna(0)
    )


    # --------------------------------------------------------
    # Cumulative flood-event count before today
    # --------------------------------------------------------

    df["historical_flood_count"] = (
        df.groupby(
            "district"
        )["flood_event"]
        .cumsum()
        .shift(1)
        .fillna(0)
    )


    # --------------------------------------------------------
    # Previous 365-day flood count
    #
    # We calculate this using dates rather than row count.
    # This is important because the data has one row per
    # district-day.
    # --------------------------------------------------------

    rolling_values = []


    for district, group in df.groupby(
        "district",
        sort=False,
    ):

        group = group.sort_values(
            "date"
        ).copy()

        flood_series = (
            group
            .set_index("date")[
                "flood_event"
            ]
        )

        previous_year_count = (
            flood_series
            .rolling(
                "365D",
                closed="left",
            )
            .sum()
        )

        group[
            "flood_count_previous_365d"
        ] = (
            previous_year_count
            .to_numpy()
        )

        rolling_values.append(
            group
        )


    df = pd.concat(
        rolling_values,
        ignore_index=True,
    )


    # --------------------------------------------------------
    # Previous 3-year flood count
    # --------------------------------------------------------

    rolling_values = []


    for district, group in df.groupby(
        "district",
        sort=False,
    ):

        group = group.sort_values(
            "date"
        ).copy()

        flood_series = (
            group
            .set_index("date")[
                "flood_event"
            ]
        )

        previous_three_year_count = (
            flood_series
            .rolling(
                "1095D",
                closed="left",
            )
            .sum()
        )

        group[
            "flood_count_previous_3y"
        ] = (
            previous_three_year_count
            .to_numpy()
        )

        rolling_values.append(
            group
        )


    df = pd.concat(
        rolling_values,
        ignore_index=True,
    )


    return df


# ============================================================
# Build complete V4 feature table
# ============================================================

def build_features(data):

    df = data.copy()


    # --------------------------------------------------------
    # Time-safe rainfall features
    # --------------------------------------------------------

    df = create_time_safe_rainfall_features(
        df
    )


    # --------------------------------------------------------
    # Time-safe historical flood features
    # --------------------------------------------------------

    df = create_flood_history_features(
        df
    )


    # --------------------------------------------------------
    # Feature list
    # --------------------------------------------------------

    feature_columns = [

        # Current rainfall
        "rainfall_24h",
        "rainfall_3day",
        "rainfall_7day",

        # Rainfall concentration
        "rainfall_24h_ratio_3day",
        "rainfall_24h_ratio_7day",
        "rainfall_3day_ratio_7day",

        # Recent rainfall
        "rainfall_recent_excess",

        # Previous rainfall
        "rainfall_previous_24h",
        "rainfall_previous_3day",
        "rainfall_previous_7day",

        # Rainfall changes
        "rainfall_24h_change",
        "rainfall_3day_change",
        "rainfall_7day_change",
        "rainfall_24h_percent_change",

        # TIME-SAFE percentile features
        "rainfall_24h_past_percentile",
        "rainfall_3day_past_percentile",
        "rainfall_7day_past_percentile",

        # Extreme rainfall
        "rainfall_24h_extreme",
        "rainfall_3day_extreme",
        "rainfall_7day_extreme",
        "rainfall_24h_very_extreme",
        "rainfall_24h_extremely_extreme",

        # Seasonality
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "monsoon_flag",

        # TIME-SAFE flood history
        "previous_flood_event",
        "historical_flood_count",
        "flood_count_previous_365d",
        "flood_count_previous_3y",
    ]


    feature_columns = [
        column
        for column in feature_columns
        if column in df.columns
    ]


    X = df[
        feature_columns
    ].copy()


    # --------------------------------------------------------
    # District identity
    # --------------------------------------------------------

    district_dummies = pd.get_dummies(
        df["district"],
        prefix="district",
        dtype=int,
    )


    X = pd.concat(
        [
            X,
            district_dummies,
        ],
        axis=1,
    )


    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )


    X = X.fillna(0)


    return (
        df,
        X,
        feature_columns,
    )


# ============================================================
# Find threshold using training data only
# ============================================================

def find_best_threshold(
    model,
    X_train,
    y_train,
):

    tscv = TimeSeriesSplit(
        n_splits=5
    )


    validation_probabilities = np.zeros(
        len(y_train)
    )


    for train_index, validation_index in tscv.split(
        X_train
    ):

        X_fold_train = X_train.iloc[
            train_index
        ]

        X_fold_validation = X_train.iloc[
            validation_index
        ]

        y_fold_train = y_train.iloc[
            train_index
        ]


        model.fit(
            X_fold_train,
            y_fold_train,
        )


        validation_probabilities[
            validation_index
        ] = model.predict_proba(
            X_fold_validation
        )[:, 1]


    thresholds = np.arange(
        0.01,
        0.901,
        0.005,
    )


    best_threshold = 0.50

    best_f2 = -1


    for threshold in thresholds:

        predictions = (
            validation_probabilities
            >= threshold
        ).astype(int)


        precision = precision_score(
            y_train,
            predictions,
            zero_division=0,
        )


        recall = recall_score(
            y_train,
            predictions,
            zero_division=0,
        )


        beta = 2


        denominator = (
            beta**2 * precision
            + recall
        )


        if denominator == 0:

            f2 = 0

        else:

            f2 = (
                (1 + beta**2)
                * precision
                * recall
                / denominator
            )


        if f2 > best_f2:

            best_f2 = f2
            best_threshold = threshold


    return (
        best_threshold,
        best_f2,
    )


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):

    print()
    print("=" * 70)

    print(
        f"MODEL: {model_name}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Threshold selection
    # --------------------------------------------------------

    print(
        "Finding threshold using training period..."
    )


    threshold, validation_f2 = (
        find_best_threshold(
            model,
            X_train,
            y_train,
        )
    )


    print(
        f"Selected threshold: {threshold:.3f}"
    )


    print(
        f"Validation F2: {validation_f2:.4f}"
    )


    # --------------------------------------------------------
    # Final model
    # --------------------------------------------------------

    print(
        "Training final model..."
    )


    model.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    predictions = (
        probabilities
        >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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


    alerts = int(
        predictions.sum()
    )


    print()
    print("TEST RESULTS")

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print(
        f"Alerts    : {alerts}"
    )

    print(
        f"Actual events: {y_test.sum()}"
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        cm
    )


    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_importances_",
    ):

        importance_table = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance":
                    model.feature_importances_,
            }
        ).sort_values(
            "importance",
            ascending=False,
        )


        print()
        print(
            "Top 15 features:"
        )


        print(
            importance_table
            .head(15)
            .to_string(
                index=False
            )
        )


    return {
        "model": model_name,
        "threshold": threshold,
        "validation_f2": validation_f2,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "test_alerts": alerts,
        "actual_test_events": int(
            y_test.sum()
        ),
        "confusion_matrix": str(
            cm.tolist()
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "FloodSense AI - Model V4 Experiment"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_data()


    print()
    print(
        f"Total observations: {len(data):,}"
    )

    print(
        f"Total flood observations: "
        f"{data[TARGET].sum():,}"
    )


    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    data, X, feature_columns = (
        build_features(data)
    )


    y = data[
        TARGET
    ].astype(int)


    # --------------------------------------------------------
    # Temporal split
    # --------------------------------------------------------

    train_mask = (
        data["date"]
        <= TRAIN_END
    )


    test_mask = (
        data["date"]
        >= TEST_START
    )


    X_train = X.loc[
        train_mask
    ].reset_index(drop=True)


    y_train = y.loc[
        train_mask
    ].reset_index(drop=True)


    X_test = X.loc[
        test_mask
    ].reset_index(drop=True)


    y_test = y.loc[
        test_mask
    ].reset_index(drop=True)


    print()
    print(
        f"Train observations: {len(X_train):,}"
    )

    print(
        f"Train flood events: {y_train.sum():,}"
    )

    print(
        f"Test observations: {len(X_test):,}"
    )

    print(
        f"Test flood events: {y_test.sum():,}"
    )


    print()
    print(
        f"Total model features: {X.shape[1]}"
    )


    print()
    print(
        "Time-safe historical features included:"
    )

    print(
        "  - Past rainfall percentiles"
    )

    print(
        "  - Previous flood event"
    )

    print(
        "  - Historical flood count"
    )

    print(
        "  - Previous 365-day flood count"
    )

    print(
        "  - Previous 3-year flood count"
    )


    # ========================================================
    # Models
    # ========================================================

    models = {

        "Random Forest V4":
            RandomForestClassifier(
                n_estimators=600,
                max_depth=10,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),


        "HistGradientBoosting V4":
            HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.04,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=42,
            ),
    }


    # ========================================================
    # Run
    # ========================================================

    results = []


    for model_name, model in models.items():

        result = evaluate_model(
            model_name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )


        results.append(
            result
        )


    # ========================================================
    # Comparison
    # ========================================================

    comparison = pd.DataFrame(
        results
    )


    comparison = comparison.sort_values(
        "test_pr_auc",
        ascending=False,
    )


    print()
    print("=" * 70)

    print(
        "MODEL V4 COMPARISON"
    )

    print("=" * 70)


    display_columns = [
        "model",
        "threshold",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "test_pr_auc",
        "test_alerts",
    ]


    print(
        comparison[
            display_columns
        ].to_string(
            index=False
        )
    )


    # ========================================================
    # Save
    # ========================================================

    comparison.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    print()
    print(
        f"Saved comparison:"
    )

    print(
        OUTPUT_FILE
    )


    print()
    print("=" * 70)

    print(
        "Model V4 experiment completed."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()