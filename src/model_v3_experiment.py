from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
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
    / "model_v3_comparison.csv"
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
            ["date", "district"]
        )
        .reset_index(drop=True)
    )

    return data


# ============================================================
# Build model features
# ============================================================

def build_features(data):

    df = data.copy()


    # --------------------------------------------------------
    # Features from V2 rainfall engineering
    # --------------------------------------------------------

    feature_columns = [

        # Original rainfall
        "rainfall_24h",
        "rainfall_3day",
        "rainfall_7day",

        # Rainfall concentration
        "rainfall_24h_ratio_3day",
        "rainfall_24h_ratio_7day",
        "rainfall_3day_ratio_7day",

        # Rainfall accumulation
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

        # Historical rainfall percentile
        "rainfall_24h_percentile",
        "rainfall_3day_percentile",
        "rainfall_7day_percentile",

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
    ]


    # --------------------------------------------------------
    # Keep only columns that actually exist
    # --------------------------------------------------------

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
    #
    # This lets the model learn that different districts
    # can have different baseline flood behaviour.
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
    # Make sure everything is numeric
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )


    X = X.fillna(0)


    return X, feature_columns


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


    # --------------------------------------------------------
    # Search thresholds
    #
    # F2 gives more importance to recall than precision.
    # This makes sense for a flood-warning prototype because
    # missing an actual flood event is important.
    # --------------------------------------------------------

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
    # Final training
    # --------------------------------------------------------

    print(
        "Training final model..."
    )


    model.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    test_predictions = (
        test_probabilities
        >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    )


    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0,
    )


    f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0,
    )


    roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )


    pr_auc = average_precision_score(
        y_test,
        test_probabilities,
    )


    cm = confusion_matrix(
        y_test,
        test_predictions,
    )


    alerts = int(
        test_predictions.sum()
    )


    actual_events = int(
        y_test.sum()
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
        f"Actual events: {actual_events}"
    )

    print()
    print("Confusion matrix:")

    print(
        cm
    )


    # --------------------------------------------------------
    # Feature importance where available
    # --------------------------------------------------------

    importances = None


    if hasattr(
        model,
        "feature_importances_",
    ):

        importances = (
            model.feature_importances_
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
        "actual_test_events": actual_events,
        "confusion_matrix": str(
            cm.tolist()
        ),
        "feature_importances": importances,
        "model_object": model,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "FloodSense AI - Model V3 Experiment"
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

    X, feature_columns = build_features(
        data
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


    # ========================================================
    # Models
    # ========================================================

    models = {

        "Random Forest V3": RandomForestClassifier(
            n_estimators=600,
            max_depth=10,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),


        "HistGradientBoosting V3":
            HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.04,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=42,
            ),


        "Logistic Regression V3":
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                C=0.5,
                solver="liblinear",
                random_state=42,
            ),
    }


    # ========================================================
    # Run experiments
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
            {
                key: value
                for key, value in result.items()
                if key
                not in [
                    "feature_importances",
                    "model_object",
                ]
            }
        )


        # ----------------------------------------------------
        # Show top feature importance
        # ----------------------------------------------------

        importances = result[
            "feature_importances"
        ]


        if importances is not None:

            importance_table = pd.DataFrame(
                {
                    "feature": X_train.columns,
                    "importance": importances,
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
                importance_table.head(15)
                .to_string(index=False)
            )


    # ========================================================
    # Comparison table
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
        "MODEL COMPARISON"
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
    # Save comparison
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
        "Model V3 experiment completed."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()