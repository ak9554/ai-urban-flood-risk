from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_high_risk_alerts.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_flood_risk_report.html"
)


def main():
    alerts = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    alerts["date"] = alerts["date"].dt.strftime(
        "%Y-%m-%d"
    )

    total_alerts = len(alerts)
    manual_reviews = int(
        alerts["manual_review_required"].sum()
    )

    district_summary = (
        alerts
        .groupby("district")
        .agg(
            alerts=("district", "size"),
            average_risk=("risk_score_percent", "mean"),
            manual_reviews=(
                "manual_review_required",
                "sum",
            ),
        )
        .reset_index()
        .sort_values("alerts", ascending=False)
    )

    district_summary["average_risk"] = (
        district_summary["average_risk"]
        .round(2)
    )

    display_columns = [
        "district",
        "date",
        "rainfall_24h",
        "rainfall_3day",
        "rainfall_7day",
        "risk_score_percent",
        "risk_level",
        "manual_review_required",
        "alert_reason",
    ]

    alert_table = (
        alerts[display_columns]
        .sort_values(
            "risk_score_percent",
            ascending=False,
        )
        .to_html(
            index=False,
            classes="alert-table",
            border=0,
        )
    )

    summary_table = district_summary.to_html(
        index=False,
        classes="summary-table",
        border=0,
    )

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NCR Flood Risk Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    background: #f4f7fb;
    color: #17202a;
}}
h1 {{
    color: #12355b;
}}
.cards {{
    display: flex;
    gap: 20px;
    margin: 25px 0;
}}
.card {{
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 8px #ccd6e0;
    min-width: 180px;
}}
.card-number {{
    font-size: 30px;
    font-weight: bold;
    color: #c0392b;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    background: white;
    margin: 20px 0 35px 0;
}}
th {{
    background: #12355b;
    color: white;
    padding: 10px;
    text-align: left;
}}
td {{
    padding: 9px;
    border-bottom: 1px solid #dde3ea;
}}
tr:hover {{
    background: #eef5fc;
}}
.note {{
    background: #fff3cd;
    border-left: 5px solid #f39c12;
    padding: 15px;
    margin: 20px 0;
}}
</style>
</head>
<body>

<h1>NCR Urban Flood Risk Report</h1>
<p>Model-generated risk alerts for the 2022–2023 evaluation period.</p>

<div class="cards">
    <div class="card">
        <div>Total high-risk alerts</div>
        <div class="card-number">{total_alerts}</div>
    </div>
    <div class="card">
        <div>Manual review alerts</div>
        <div class="card-number">{manual_reviews}</div>
    </div>
    <div class="card">
        <div>Districts covered</div>
        <div class="card-number">
            {alerts["district"].nunique()}
        </div>
    </div>
</div>

<div class="note">
<strong>Important:</strong>
These are model-generated prototype alerts, not confirmed flood warnings.
Rows marked for manual review have high model risk but zero recent rainfall.
</div>

<h2>Alert summary by district</h2>
{summary_table}

<h2>High-risk alerts</h2>
{alert_table}

</body>
</html>
"""

    OUTPUT_FILE.write_text(
        html,
        encoding="utf-8",
    )

    print("=" * 60)
    print("NCR flood-risk HTML report created")
    print("=" * 60)
    print()
    print(f"Total alerts: {total_alerts}")
    print(f"Manual reviews: {manual_reviews}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()