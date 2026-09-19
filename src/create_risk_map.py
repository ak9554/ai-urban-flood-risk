from pathlib import Path
import json

import folium
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ALERT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_risk_predictions_2022_2023.csv"
)

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_7_districts.geojson"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ncr_flood_risk_map.html"
)


def risk_color(value):
    if value >= 20:
        return "#800026"
    if value >= 10:
        return "#E31A1C"
    if value >= 5:
        return "#FD8D3C"
    return "#31A354"


def main():
    predictions = pd.read_csv(
        ALERT_FILE,
        parse_dates=["date"],
    )

    summary = (
        predictions
        .groupby("district")
        .agg(
            average_risk=("risk_score_percent", "mean"),
            maximum_risk=("risk_score_percent", "max"),
            alerts=("flood_alert", "sum"),
            flood_events=("flood_event", "sum"),
        )
        .reset_index()
    )

    with open(
        BOUNDARY_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        geojson = json.load(file)

    summary_by_district = (
        summary
        .set_index("district")
        .to_dict("index")
    )

    for feature in geojson["features"]:
        district = feature["properties"]["DISTRICT"]
        values = summary_by_district.get(
            district,
            {
                "average_risk": 0,
                "maximum_risk": 0,
                "alerts": 0,
                "flood_events": 0,
            },
        )

        feature["properties"]["average_risk"] = round(
            values["average_risk"],
            2,
        )
        feature["properties"]["maximum_risk"] = round(
            values["maximum_risk"],
            2,
        )
        feature["properties"]["alerts"] = int(
            values["alerts"]
        )
        feature["properties"]["flood_events"] = int(
            values["flood_events"]
        )

    risk_map = folium.Map(
        location=[28.50, 77.10],
        zoom_start=9,
        tiles="CartoDB positron",
    )

    folium.GeoJson(
        geojson,
        name="NCR flood risk",
        style_function=lambda feature: {
            "fillColor": risk_color(
                feature["properties"]["average_risk"]
            ),
            "color": "#222222",
            "weight": 1,
            "fillOpacity": 0.65,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "DISTRICT",
                "average_risk",
                "maximum_risk",
                "alerts",
                "flood_events",
            ],
            aliases=[
                "District",
                "Average risk (%)",
                "Maximum risk (%)",
                "Alerts",
                "Observed flood events",
            ],
            localize=True,
            sticky=False,
        ),
    ).add_to(risk_map)

    folium.LayerControl().add_to(risk_map)

    risk_map.save(OUTPUT_FILE)

    print("=" * 60)
    print("NCR flood-risk map created")
    print("=" * 60)
    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()