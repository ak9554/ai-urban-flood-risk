from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_risk_model_training.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "ncr_flood_risk_random_forest.joblib"
)

TARGET_COLUMN = "flood_event"
DATE_COLUMN = "date"
DISTRICT_COLUMN = "district"

BASE_FEATURES = [
    "rainfall_24h",
    "rainfall_3day",
    "rainfall_7day",
    "historical_flooded_area_percent",
    "population",
    "historical_mean_flood_duration",
]

TRAIN_END_DATE = pd.Timestamp("2021-12-31")
TEST_START_DATE = pd.Timestamp("2022-01-01")
TEST_END_DATE = pd.Timestamp("2023-12-31")


def create_model():
    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def build_features(data):
    result = data[BASE_FEATURES].copy()

    dates = pd.to_datetime(data[DATE_COLUMN])

    month = dates.dt.month
    day_of_year = dates.dt.dayofyear

    result["month_sin"] = np.sin(2 * np.pi * month / 12)
    result["month_cos"] = np.cos(2 * np.pi * month / 12)

    result["day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )
    result["day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    result["rainfall_24h_ratio_7day"] = (
        data["rainfall_24h"]
        / data["rainfall_7day"].replace(0, np.nan)
    )

    result["rainfall_3day_ratio_7day"] = (
        data["rainfall_3day"]
        / data["rainfall_7day"].replace(0, np.nan)
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)

    district_dummies = pd.get_dummies(
        data[DISTRICT_COLUMN],
        prefix="district",
        dtype=int,
    )

    return pd.concat(
        [result.reset_index(drop=True), district_dummies],
        axis=1,
    )


def find_best_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    precision = precision[:-1]
    recall = recall[:-1]

    f2_scores = (
        5 * precision * recall
        / ((4 * precision) + recall + 1e-12)
    )

    best_index = int(np.nanargmax(f2_scores))

    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f2": float(f2_scores[best_index]),
    }


def main():
    print("=" * 60)
    print("Training improved NCR flood-risk Random Forest model")
    print("=" * 60)

    data = pd.read_csv(
        TRAINING_FILE,
        parse_dates=[DATE_COLUMN],
    )

    required_columns = set(
        BASE_FEATURES
        + [
            DATE_COLUMN,
            DISTRICT_COLUMN,
            TARGET_COLUMN,
        ]
    )

    missing_columns = sorted(
        required_columns.difference(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
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
        (data[DATE_COLUMN] >= TEST_START_DATE)
        & (data[DATE_COLUMN] <= TEST_END_DATE)
    ].copy()

    X_all = build_features(data)

    X_train = X_all.loc[train.index]
    X_test = X_all.loc[test.index]

    y_train = train[TARGET_COLUMN].astype(int)
    y_test = test[TARGET_COLUMN].astype(int)

    print()
    print("Dataset summary:")
    print(f"Total rows: {len(data):,}")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Training flood events: {int(y_train.sum()):,}")
    print(f"Test flood events: {int(y_test.sum()):,}")
    print(f"Model features: {X_train.shape[1]}")

    print()
    print("Finding threshold using training data only...")

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    out_of_fold_probabilities = cross_val_predict(
        create_model(),
        X_train,
        y_train,
        cv=cross_validation,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    threshold_result = find_best_threshold(
        y_train,
        out_of_fold_probabilities,
    )

    selected_threshold = threshold_result["threshold"]

    model = create_model()
    model.fit(X_train, y_train)

    test_probabilities = model.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_probabilities >= selected_threshold
    ).astype(int)

    print()
    print("Selected threshold:")
    print(f"Threshold: {selected_threshold:.6f}")
    print(
        f"Training CV precision: "
        f"{threshold_result['precision']:.4f}"
    )
    print(
        f"Training CV recall: "
        f"{threshold_result['recall']:.4f}"
    )
    print(
        f"Training CV F2: "
        f"{threshold_result['f2']:.4f}"
    )

    print()
    print("Test metrics:")
    print(
        f"Precision: "
        f"{precision_score(y_test, test_predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall:    "
        f"{recall_score(y_test, test_predictions, zero_division=0):.4f}"
    )
    print(
        f"F1 score:  "
        f"{f1_score(y_test, test_predictions, zero_division=0):.4f}"
    )

    if y_test.nunique() == 2:
        print(
            f"ROC-AUC:   "
            f"{roc_auc_score(y_test, test_probabilities):.4f}"
        )
        print(
            f"PR-AUC:    "
            f"{average_precision_score(y_test, test_probabilities):.4f}"
        )

    print()
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(
        confusion_matrix(
            y_test,
            test_predictions,
            labels=[0, 1],
        )
    )

    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    print()
    print("Top feature importance:")

    importances = pd.Series(
        model.feature_importances_,
        index=X_train.columns,
    )

    print(
        importances
        .sort_values(ascending=False)
        .head(15)
        .to_string()
    )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "features": X_train.columns.tolist(),
            "threshold": selected_threshold,
            "threshold_metric": "F2",
            "train_end_date": str(
                TRAIN_END_DATE.date()
            ),
        },
        MODEL_FILE,
    )

    print()
    print(f"Saved model: {MODEL_FILE}")
    print("=" * 60)
    print("Improved model training completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()