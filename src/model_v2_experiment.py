from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_risk_model_training.csv"
)


# ============================================================
# Columns
# ============================================================

TARGET_COLUMN = "flood_event"
DATE_COLUMN = "date"
DISTRICT_COLUMN = "district"


BASE_FEATURES = [
    "rainfall_24h",
    "rainfall_3day",
    "rainfall_7day",
]


HISTORICAL_CONTEXT_FEATURES = [
    "historical_flooded_area_percent",
    "population",
    "historical_mean_flood_duration",
]


TRAIN_END_DATE = pd.Timestamp("2021-12-31")

TEST_START_DATE = pd.Timestamp("2022-01-01")

TEST_END_DATE = pd.Timestamp("2023-12-31")


# ============================================================
# Feature engineering
# ============================================================

def build_features(data, include_historical_context=False):
    """
    Build model features.

    The model uses rainfall, seasonal information,
    rainfall concentration ratios, and district identity.

    Historical context can optionally be included so that
    we can compare whether it improves the temporal test.
    """

    result = data[BASE_FEATURES].copy()

    dates = pd.to_datetime(data[DATE_COLUMN])

    month = dates.dt.month

    day_of_year = dates.dt.dayofyear


    # --------------------------------------------------------
    # Seasonal features
    # --------------------------------------------------------

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
    # Rainfall concentration
    # --------------------------------------------------------

    result["rainfall_24h_ratio_7day"] = (
        data["rainfall_24h"]
        / data["rainfall_7day"].replace(0, np.nan)
    )

    result["rainfall_3day_ratio_7day"] = (
        data["rainfall_3day"]
        / data["rainfall_7day"].replace(0, np.nan)
    )


    # --------------------------------------------------------
    # Additional rainfall intensity indicators
    # --------------------------------------------------------

    result["rainfall_24h_per_day_3day"] = (
        data["rainfall_24h"]
        / 3.0
    )

    result["rainfall_3day_per_day"] = (
        data["rainfall_3day"]
        / 3.0
    )

    result["rainfall_7day_per_day"] = (
        data["rainfall_7day"]
        / 7.0
    )


    # --------------------------------------------------------
    # Optional historical context
    # --------------------------------------------------------

    if include_historical_context:

        for column in HISTORICAL_CONTEXT_FEATURES:

            result[column] = data[column]


    # --------------------------------------------------------
    # Clean numerical values
    # --------------------------------------------------------

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = result.fillna(0)


    # --------------------------------------------------------
    # District identity
    # --------------------------------------------------------

    district_dummies = pd.get_dummies(
        data[DISTRICT_COLUMN],
        prefix="district",
        dtype=int,
    )


    result = pd.concat(
        [
            result.reset_index(drop=True),
            district_dummies.reset_index(drop=True),
        ],
        axis=1,
    )


    return result


# ============================================================
# Models
# ============================================================

def create_random_forest():

    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def create_hist_gradient_boosting():

    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )


def create_logistic_regression():

    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=0.5,
        random_state=42,
    )


# ============================================================
# Time-aware threshold search
# ============================================================

def find_best_threshold(y_true, probabilities):

    thresholds = np.linspace(
        0.01,
        0.90,
        180,
    )

    best_threshold = 0.50

    best_f2 = -1

    best_precision = 0

    best_recall = 0


    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)


        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )


        # F2 gives recall more importance than precision.
        f2 = (
            5 * precision * recall
            / (
                4 * precision
                + recall
                + 1e-12
            )
        )


        if f2 > best_f2:

            best_f2 = f2

            best_threshold = threshold

            best_precision = precision

            best_recall = recall


    return {
        "threshold": best_threshold,
        "precision": best_precision,
        "recall": best_recall,
        "f2": best_f2,
    }


# ============================================================
# Time-series validation for threshold selection
# ============================================================

def get_oof_probabilities(
    model_factory,
    X,
    y,
):

    splitter = TimeSeriesSplit(
        n_splits=5
    )


    probabilities = np.full(
        len(y),
        np.nan,
        dtype=float,
    )


    for train_indices, validation_indices in splitter.split(X):

        X_train_fold = X.iloc[train_indices]

        X_validation_fold = X.iloc[
            validation_indices
        ]

        y_train_fold = y.iloc[
            train_indices
        ]


        model = model_factory()

        model.fit(
            X_train_fold,
            y_train_fold,
        )


        probabilities[
            validation_indices
        ] = model.predict_proba(
            X_validation_fold
        )[:, 1]


    valid = ~np.isnan(probabilities)

    return (
        probabilities[valid],
        y.iloc[np.where(valid)[0]],
    )


# ============================================================
# Evaluate one model
# ============================================================

def evaluate_model(
    model_name,
    model_factory,
    X_train,
    y_train,
    X_test,
    y_test,
):

    print()
    print("-" * 60)
    print(f"MODEL: {model_name}")
    print("-" * 60)


    print("Running time-aware validation...")


    validation_probabilities, validation_targets = (
        get_oof_probabilities(
            model_factory,
            X_train,
            y_train,
        )
    )


    threshold_result = find_best_threshold(
        validation_targets,
        validation_probabilities,
    )


    threshold = threshold_result[
        "threshold"
    ]


    print(
        f"Validation threshold: "
        f"{threshold:.3f}"
    )

    print(
        f"Validation precision: "
        f"{threshold_result['precision']:.4f}"
    )

    print(
        f"Validation recall: "
        f"{threshold_result['recall']:.4f}"
    )

    print(
        f"Validation F2: "
        f"{threshold_result['f2']:.4f}"
    )


    print()
    print("Training final model...")


    model = model_factory()

    model.fit(
        X_train,
        y_train,
    )


    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]


    test_predictions = (
        test_probabilities >= threshold
    ).astype(int)


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


    print()
    print("TEST RESULTS")

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:    {pr_auc:.4f}"
    )


    print()
    print(
        "Confusion matrix "
        "[[TN, FP], [FN, TP]]:"
    )

    print(
        confusion_matrix(
            y_test,
            test_predictions,
            labels=[0, 1],
        )
    )


    print()
    print(
        f"Predicted flood events: "
        f"{int(test_predictions.sum())}"
    )

    print(
        f"Actual flood events: "
        f"{int(y_test.sum())}"
    )


    return {
        "model": model_name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "predicted_alerts": int(
            test_predictions.sum()
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)

    print(
        "FloodSense AI - Model V2 Experiment"
    )

    print("=" * 60)


    data = pd.read_csv(
        TRAINING_FILE,
        parse_dates=[DATE_COLUMN],
    )


    data = (
        data
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )


    train = data[
        data[DATE_COLUMN] <= TRAIN_END_DATE
    ].copy()


    test = data[
        (
            data[DATE_COLUMN]
            >= TEST_START_DATE
        )
        &
        (
            data[DATE_COLUMN]
            <= TEST_END_DATE
        )
    ].copy()


    print()
    print("DATASET")

    print(
        f"Total rows: "
        f"{len(data):,}"
    )

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Test rows: "
        f"{len(test):,}"
    )

    print(
        f"Training flood events: "
        f"{int(train[TARGET_COLUMN].sum()):,}"
    )

    print(
        f"Test flood events: "
        f"{int(test[TARGET_COLUMN].sum()):,}"
    )


    # --------------------------------------------------------
    # Experiment 1
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        "EXPERIMENT 1: "
        "Rainfall + seasonal + district"
    )

    print("=" * 60)


    X = build_features(
        data,
        include_historical_context=False,
    )


    X_train = X.loc[train.index]

    X_test = X.loc[test.index]


    y_train = train[
        TARGET_COLUMN
    ].astype(int)

    y_test = test[
        TARGET_COLUMN
    ].astype(int)


    results = []


    results.append(
        evaluate_model(
            "Random Forest - Core",
            create_random_forest,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )


    results.append(
        evaluate_model(
            "HistGradientBoosting - Core",
            create_hist_gradient_boosting,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )


    results.append(
        evaluate_model(
            "Logistic Regression - Core",
            create_logistic_regression,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )


    # --------------------------------------------------------
    # Experiment 2
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        "EXPERIMENT 2: "
        "Core + historical context"
    )

    print("=" * 60)


    X_context = build_features(
        data,
        include_historical_context=True,
    )


    X_context_train = X_context.loc[
        train.index
    ]

    X_context_test = X_context.loc[
        test.index
    ]


    results.append(
        evaluate_model(
            "Random Forest - Context",
            create_random_forest,
            X_context_train,
            y_train,
            X_context_test,
            y_test,
        )
    )


    results.append(
        evaluate_model(
            "HistGradientBoosting - Context",
            create_hist_gradient_boosting,
            X_context_train,
            y_train,
            X_context_test,
            y_test,
        )
    )


    results.append(
        evaluate_model(
            "Logistic Regression - Context",
            create_logistic_regression,
            X_context_train,
            y_train,
            X_context_test,
            y_test,
        )
    )


    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        results
    )


    print()
    print("=" * 60)

    print("MODEL COMPARISON")

    print("=" * 60)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    output_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "model_v2_comparison.csv"
    )


    comparison.to_csv(
        output_file,
        index=False,
    )


    print()
    print(
        f"Saved comparison: "
        f"{output_file}"
    )


    print()
    print("=" * 60)

    print("Model V2 experiment completed.")

    print("=" * 60)


if __name__ == "__main__":

    main()