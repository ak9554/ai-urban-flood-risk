from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_risk_predictions_2022_2023.csv"
)

DATE_COLUMN = "date"
TARGET_COLUMN = "flood_event"

BASE_FEATURES = [
    "rainfall_24h",
    "rainfall_3day",
    "rainfall_7day",
    "historical_flooded_area_percent",
    "population",
    "historical_mean_flood_duration",
]


def build_features(data):
    result = data[BASE_FEATURES].copy()

    dates = pd.to_datetime(data[DATE_COLUMN])

    month = dates.dt.month
    day_of_year = dates.dt.dayofyear

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
        data["district"],
        prefix="district",
        dtype=int,
    )

    return pd.concat(
        [
            result.reset_index(drop=True),
            district_dummies.reset_index(drop=True),
        ],
        axis=1,
    )


def assign_risk_level(score):
    if score >= 0.20:
        return "VERY_HIGH"
    if score >= 0.10:
        return "HIGH"
    if score >= 0.05:
        return "MEDIUM"
    return "LOW"


def main():
    print("=" * 60)
    print("Creating NCR flood-risk predictions")
    print("=" * 60)

    data = pd.read_csv(
        INPUT_FILE,
        parse_dates=[DATE_COLUMN],
    )

    saved_model = joblib.load(MODEL_FILE)

    model = saved_model["model"]
    model_features = saved_model["features"]
    alert_threshold = saved_model["threshold"]

    features = build_features(data)

    features = features.reindex(
        columns=model_features,
        fill_value=0,
    )

    probabilities = model.predict_proba(features)[:, 1]

    data["risk_score"] = probabilities
    data["risk_score_percent"] = (
        probabilities * 100
    ).round(2)

    data["risk_level"] = [
        assign_risk_level(score)
        for score in probabilities
    ]

    data["flood_alert"] = (
        probabilities >= alert_threshold
    ).astype(int)

    predictions = data[
        (data[DATE_COLUMN] >= "2022-01-01")
        & (data[DATE_COLUMN] <= "2023-12-31")
    ].copy()

    output_columns = [
        "district",
        "date",
        "rainfall_24h",
        "rainfall_3day",
        "rainfall_7day",
        "risk_score",
        "risk_score_percent",
        "risk_level",
        "flood_alert",
        "flood_event",
    ]

    predictions = predictions[output_columns]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(f"Rows exported: {len(predictions):,}")
    print()
    print("Risk-level distribution:")
    print(
        predictions["risk_level"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Flood-alert distribution:")
    print(
        predictions["flood_alert"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Highest-risk predictions:")
    print(
        predictions
        .sort_values("risk_score", ascending=False)
        .head(15)
        .to_string(index=False)
    )

    predictions.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(f"Saved predictions: {OUTPUT_FILE}")
    print("=" * 60)
    print("Risk prediction export completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()