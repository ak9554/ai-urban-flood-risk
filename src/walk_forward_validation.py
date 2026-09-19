from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
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
    / "walk_forward_validation_v2.csv"
)


# ============================================================
# Configuration
# ============================================================

TARGET = "flood_event"


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

    rainfall_columns = [
        "rainfall_24h",
        "rainfall_3day",
        "rainfall_7day",
    ]

    for column in rainfall_columns:

        output_column = (
            column
            + "_past_percentile"
        )

        results = []

        for district, group in df.groupby(
            "district",
            sort=False,
        ):

            group = group.sort_values(
                "date"
            ).copy()

            rainfall = (
                group[column]
                .astype(float)
                .to_numpy()
            )

            percentiles = []

            for i in range(
                len(rainfall)
            ):

                if i == 0:

                    percentiles.append(
                        0.5
                    )

                else:

                    previous_values = (
                        rainfall[:i]
                    )

                    percentile = np.mean(
                        previous_values
                        <= rainfall[i]
                    )

                    percentiles.append(
                        percentile
                    )

            group[
                output_column
            ] = percentiles

            results.append(
                group[
                    [
                        "district",
                        "date",
                        output_column,
                    ]
                ]
            )

        percentile_table = pd.concat(
            results,
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

    df = (
        df
        .sort_values(
            [
                "district",
                "date",
            ]
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Was the previous day a flood day?
    # --------------------------------------------------------

    df["previous_flood_event"] = (
        df.groupby(
            "district"
        )["flood_event"]
        .shift(1)
        .fillna(0)
    )


    # --------------------------------------------------------
    # Number of flood days before today
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
    # --------------------------------------------------------

    results = []

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

        previous_year = (
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
            previous_year
            .to_numpy()
        )

        results.append(
            group
        )


    df = pd.concat(
        results,
        ignore_index=True,
    )


    # --------------------------------------------------------
    # Previous 3-year flood count
    # --------------------------------------------------------

    results = []

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

        previous_three_years = (
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
            previous_three_years
            .to_numpy()
        )

        results.append(
            group
        )


    df = pd.concat(
        results,
        ignore_index=True,
    )


    return df


# ============================================================
# Build all time-safe features
# ============================================================

def create_features(data):

    df = create_time_safe_rainfall_features(
        data
    )

    df = create_flood_history_features(
        df
    )

    return df


# ============================================================
# Build feature matrix
# ============================================================

def build_feature_matrix(data):

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

        # Rainfall change
        "rainfall_24h_change",
        "rainfall_3day_change",
        "rainfall_7day_change",
        "rainfall_24h_percent_change",

        # Time-safe percentiles
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

        # Historical flood information
        "previous_flood_event",
        "historical_flood_count",
        "flood_count_previous_365d",
        "flood_count_previous_3y",
    ]


    feature_columns = [
        column
        for column in feature_columns
        if column in data.columns
    ]


    X = data[
        feature_columns
    ].copy()


    # --------------------------------------------------------
    # District identity
    # --------------------------------------------------------

    district_dummies = pd.get_dummies(
        data["district"],
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


    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )


    X = X.fillna(0)


    return X


# ============================================================
# Find threshold using chronological TimeSeriesSplit
# ============================================================

def find_best_threshold(
    X_train,
    y_train,
):

    print(
        "Finding threshold using "
        "chronological validation..."
    )


    tscv = TimeSeriesSplit(
        n_splits=5
    )


    validation_probabilities = np.full(
        len(y_train),
        np.nan,
    )


    validation_mask = np.zeros(
        len(y_train),
        dtype=bool,
    )


    for fold_number, (
        train_index,
        validation_index,
    ) in enumerate(
        tscv.split(X_train),
        start=1,
    ):

        X_fold_train = X_train.iloc[
            train_index
        ]

        y_fold_train = y_train.iloc[
            train_index
        ]

        X_fold_validation = X_train.iloc[
            validation_index
        ]

        y_fold_validation = y_train.iloc[
            validation_index
        ]


        print(
            f"  Fold {fold_number}: "
            f"train={len(train_index):,}, "
            f"validation={len(validation_index):,}, "
            f"validation events="
            f"{y_fold_validation.sum()}"
        )


        model = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        )


        model.fit(
            X_fold_train,
            y_fold_train,
        )


        validation_probabilities[
            validation_index
        ] = model.predict_proba(
            X_fold_validation
        )[:, 1]


        validation_mask[
            validation_index
        ] = True


    # --------------------------------------------------------
    # Keep only observations that actually participated in
    # validation.
    # --------------------------------------------------------

    probabilities = (
        validation_probabilities[
            validation_mask
        ]
    )

    actual = (
        y_train[
            validation_mask
        ]
    )


    print()
    print(
        f"Total validation observations: "
        f"{len(actual):,}"
    )

    print(
        f"Total validation flood events: "
        f"{actual.sum()}"
    )


    # --------------------------------------------------------
    # Search thresholds
    # --------------------------------------------------------

    thresholds = np.arange(
        0.005,
        0.901,
        0.005,
    )


    best_threshold = 0.50

    best_f2 = -1


    for threshold in thresholds:

        predictions = (
            probabilities
            >= threshold
        ).astype(int)


        precision = precision_score(
            actual,
            predictions,
            zero_division=0,
        )


        recall = recall_score(
            actual,
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
# Evaluate one test year
# ============================================================

def evaluate_year(
    data,
    X,
    test_year,
):

    print()
    print("=" * 70)

    print(
        f"TEST YEAR: {test_year}"
    )

    print("=" * 70)


    train_mask = (
        data["date"].dt.year
        < test_year
    )


    test_mask = (
        data["date"].dt.year
        == test_year
    )


    X_train = X.loc[
        train_mask
    ].reset_index(drop=True)


    y_train = data.loc[
        train_mask,
        TARGET,
    ].astype(int).reset_index(
        drop=True
    )


    X_test = X.loc[
        test_mask
    ].reset_index(drop=True)


    y_test = data.loc[
        test_mask,
        TARGET,
    ].astype(int).reset_index(
        drop=True
    )


    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Training flood events: "
        f"{y_train.sum()}"
    )

    print(
        f"Test rows: {len(X_test):,}"
    )

    print(
        f"Test flood events: "
        f"{y_test.sum()}"
    )


    # --------------------------------------------------------
    # Find threshold
    # --------------------------------------------------------

    threshold, validation_f2 = (
        find_best_threshold(
            X_train,
            y_train,
        )
    )


    print()
    print(
        f"Selected threshold: "
        f"{threshold:.3f}"
    )

    print(
        f"Validation F2: "
        f"{validation_f2:.4f}"
    )


    # --------------------------------------------------------
    # Train final model on all available historical data
    # --------------------------------------------------------

    final_model = (
        HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        )
    )


    final_model.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    probabilities = (
        final_model
        .predict_proba(
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


    if y_test.nunique() == 2:

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )

        pr_auc = average_precision_score(
            y_test,
            probabilities,
        )

    else:

        roc_auc = np.nan
        pr_auc = np.nan


    cm = confusion_matrix(
        y_test,
        predictions,
    )


    alerts = int(
        predictions.sum()
    )


    print()
    print(
        "TEST RESULTS"
    )

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


    print()
    print(
        "Confusion matrix:"
    )

    print(
        cm
    )


    return {
        "test_year": test_year,
        "training_rows": len(X_train),
        "training_events": int(
            y_train.sum()
        ),
        "test_rows": len(X_test),
        "test_events": int(
            y_test.sum()
        ),
        "threshold": threshold,
        "validation_f2": validation_f2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "alerts": alerts,
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
        "FloodSense AI - Walk-Forward Validation V2"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_data()


    print()
    print(
        f"Total observations: "
        f"{len(data):,}"
    )

    print(
        f"Total flood events: "
        f"{data[TARGET].sum()}"
    )


    # --------------------------------------------------------
    # Create features
    # --------------------------------------------------------

    print()
    print(
        "Creating time-safe features..."
    )


    data = create_features(
        data
    )


    X = build_feature_matrix(
        data
    )


    print(
        f"Total model features: "
        f"{X.shape[1]}"
    )


    # --------------------------------------------------------
    # Test years
    # --------------------------------------------------------

    test_years = [
        2019,
        2020,
        2021,
        2022,
        2023,
    ]


    results = []


    for year in test_years:

        result = evaluate_year(
            data,
            X,
            year,
        )

        results.append(
            result
        )


    # ========================================================
    # Summary
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    print()
    print("=" * 70)

    print(
        "WALK-FORWARD VALIDATION V2 SUMMARY"
    )

    print("=" * 70)


    print(
        results_df[
            [
                "test_year",
                "test_events",
                "threshold",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "alerts",
            ]
        ].to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Averages
    # --------------------------------------------------------

    print()
    print(
        "Average metrics:"
    )


    print(
        f"Mean precision: "
        f"{results_df['precision'].mean():.4f}"
    )


    print(
        f"Mean recall: "
        f"{results_df['recall'].mean():.4f}"
    )


    print(
        f"Mean F1: "
        f"{results_df['f1'].mean():.4f}"
    )


    print(
        f"Mean ROC-AUC: "
        f"{results_df['roc_auc'].mean():.4f}"
    )


    print(
        f"Mean PR-AUC: "
        f"{results_df['pr_auc'].mean():.4f}"
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    print()
    print(
        "Saved validation results:"
    )

    print(
        OUTPUT_FILE
    )


    print()
    print("=" * 70)

    print(
        "Walk-forward validation V2 completed."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()