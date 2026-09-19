from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_risk_predictions_2022_2023.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_high_risk_alerts.csv"
)


def main():
    predictions = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    alerts = predictions[
        predictions["risk_score"] >= 0.10
    ].copy()

    alerts["manual_review_required"] = (
        (alerts["rainfall_7day"] == 0)
        & (alerts["risk_score"] >= 0.20)
    ).astype(int)

    alerts["alert_reason"] = alerts.apply(
        lambda row: (
            "High model score but zero recent rainfall"
            if row["manual_review_required"] == 1
            else "High rainfall-related model risk"
        ),
        axis=1,
    )

    alerts = alerts.sort_values(
        "risk_score",
        ascending=False,
    )

    output_columns = [
        "district",
        "date",
        "rainfall_24h",
        "rainfall_3day",
        "rainfall_7day",
        "risk_score_percent",
        "risk_level",
        "flood_alert",
        "manual_review_required",
        "alert_reason",
        "flood_event",
    ]

    alerts[output_columns].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 60)
    print("High-risk flood alerts exported")
    print("=" * 60)
    print()
    print("Total alerts:", len(alerts))
    print(
        "Alerts requiring manual review:",
        int(alerts["manual_review_required"].sum()),
    )

    print()
    print("Alerts by district:")
    print(
        alerts["district"]
        .value_counts()
        .to_string()
    )

    print()
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()