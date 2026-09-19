from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_risk_predictions_2022_2023.csv"
)


def main():
    predictions = pd.read_csv(
        PREDICTION_FILE,
        parse_dates=["date"],
    )

    print("=" * 60)
    print("Checking NCR flood-risk predictions")
    print("=" * 60)

    print()
    print("Rows:", len(predictions))

    print()
    print("Actual flood events by district:")
    print(
        predictions
        .groupby("district")["flood_event"]
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    print()
    print("Average risk score by district:")
    print(
        predictions
        .groupby("district")["risk_score_percent"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
        .to_string()
    )

    print()
    print("Flood-alert count by district:")
    print(
        predictions
        .groupby("district")["flood_alert"]
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    suspicious = predictions[
        (predictions["risk_score"] >= 0.20)
        & (predictions["rainfall_7day"] == 0)
    ].copy()

    print()
    print(
        "Very-high-risk predictions with zero 7-day rainfall:",
        len(suspicious),
    )

    if not suspicious.empty:
        suspicious_display = suspicious[
            [
                "district",
                "date",
                "rainfall_24h",
                "rainfall_3day",
                "rainfall_7day",
                "risk_score",
                "risk_score_percent",
                "risk_level",
                "flood_event",
            ]
        ].sort_values(
            "risk_score",
            ascending=False,
        )

        print(
            suspicious_display
            .head(20)
            .to_string(index=False)
        )

    print()
    print("Top 20 predictions:")

    top_predictions = predictions[
        [
            "district",
            "date",
            "risk_score",
            "risk_score_percent",
            "risk_level",
            "flood_alert",
            "flood_event",
        ]
    ].sort_values(
        "risk_score",
        ascending=False,
    )

    print(
        top_predictions
        .head(20)
        .to_string(index=False)
    )

    print()
    print("=" * 60)
    print("Prediction quality check completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()