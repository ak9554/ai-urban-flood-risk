from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import folium
    from folium.plugins import Fullscreen, MarkerCluster
    from streamlit_folium import st_folium
except Exception:
    folium = None
    Fullscreen = MarkerCluster = st_folium = None

try:
    import plotly.express as px
except Exception:
    px = None

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
except Exception:
    HistGradientBoostingClassifier = None
    balanced_accuracy_score = f1_score = precision_score = recall_score = None


# FloodSense AI — complete Streamlit application.
# It reads the project's processed outputs, normalizes aliases once, and keeps
# implementation vocabulary out of the public-facing experience.

def find_project_root() -> Path:
    candidates = [Path(__file__).resolve().parents[1], Path.cwd()]
    candidates.extend(Path(__file__).resolve().parents[i] for i in range(2, 5))
    for candidate in candidates:
        if (candidate / "data" / "processed" / "ncr_hyperlocal_risk.csv").exists():
            return candidate
    return Path(__file__).resolve().parents[1]


ROOT = find_project_root()
DATA_DIR = ROOT / "data" / "processed"
RISK_FILE = DATA_DIR / "ncr_hyperlocal_risk.csv"
BOUNDARY_FILE = DATA_DIR / "ncr_7_districts.geojson"
HISTORY_FILE = DATA_DIR / "ncr_historical_context.csv"
BIHAR_FLOODED_AREA_FILE = ROOT / "data" / "raw" / "District_FloodedArea.csv"
BIHAR_IMPACT_FILE = ROOT / "data" / "raw" / "District_FloodImpact.csv"
BIHAR_INVENTORY_FILE = ROOT / "data" / "raw" / "India_Flood_Inventory_v3.csv"
MAP_POINT_LIMIT = 500
SCENARIO_POINT_LIMIT = 700

PAGES = [
    "Command Center",
    "Risk Map",
    "Why This Area?",
    "Intervention Priority",
    "Response Planner",
    "Scenario Lab",
    "Compare Districts",
    "Bihar Validation",
    "Methodology & Model Insights",
]

PAGE_ICONS = {
    "Command Center": "⌂",
    "Risk Map": "◉",
    "Why This Area?": "✦",
    "Intervention Priority": "▣",
    "Response Planner": "↗",
    "Scenario Lab": "◇",
    "Compare Districts": "⇄",
    "Bihar Validation": "◆",
    "Methodology & Model Insights": "◌",
}

PAGE_LABELS = {
    "Command Center": "Command Center",
    "Risk Map": "Flood Map",
    "Why This Area?": "Why This Area?",
    "Intervention Priority": "What Should We Check First?",
    "Response Planner": "Response Planner",
    "Scenario Lab": "Rainfall What-If",
    "Compare Districts": "Compare Districts",
    "Bihar Validation": "Bihar Evidence",
    "Methodology & Model Insights": "Methodology & Model Insights",
}

RISK_COLORS = {
    "Very Low": "#3b82f6",
    "Low": "#22c55e",
    "Moderate": "#f59e0b",
    "High": "#f97316",
    "Very High": "#ef4444",
}


def first_column(df: pd.DataFrame, names: list[str], default: Any = np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def minmax(series: pd.Series) -> pd.Series:
    values = numeric(series)
    lo, hi = float(values.min()), float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(0.0, index=series.index)
    return (values - lo) / (hi - lo) * 100


def classify(score: float) -> str:
    if score >= 80:
        return "Very High"
    if score >= 60:
        return "High"
    if score >= 40:
        return "Moderate"
    if score >= 20:
        return "Low"
    return "Very Low"


def priority_for(score: float) -> str:
    if score >= 60:
        return "Immediate Review"
    if score >= 40:
        return "High Priority"
    if score >= 20:
        return "Monitor"
    return "Routine Monitoring"


def friendly_location(value: Any) -> str:
    text = str(value)
    if text.startswith("NCR_"):
        return f"Area {text.removeprefix('NCR_')}"
    return text


def friendly_priority(value: Any) -> str:
    return {
        "Routine Monitoring": "Routine check",
        "Monitor": "Keep watch",
        "High Priority": "High attention",
        "Immediate Review": "Review now",
    }.get(str(value), str(value))


def friendly_driver(value: Any) -> str:
    return {
        "Slope/runoff": "Ground shape and runoff",
        "Built-up land": "Built-up surfaces",
        "High historical vulnerability context": "Past flood impact",
        "Elevated 24-hour rainfall": "Recent heavy rain",
        "High 24-hour rainfall": "Very heavy recent rain",
        "Elevated temporal model signal": "Recent weather pattern",
        "Moderate temporal model signal": "Moderate recent weather pattern",
    }.get(str(value), str(value).replace("_", " "))


def friendly_explanation(value: Any) -> str:
    text = str(value)
    replacements = {
        "Slope/runoff": "Ground shape and runoff",
        "Built-up land": "Built-up surfaces",
        "High historical vulnerability context": "Past flood impact",
        "Elevated 24-hour rainfall": "Recent heavy rain",
        "High 24-hour rainfall": "Very heavy recent rain",
        "Elevated temporal model signal": "Recent weather pattern",
        "Moderate temporal model signal": "Moderate recent weather pattern",
    }
    for technical, plain in replacements.items():
        text = text.replace(technical, plain)
    return text.replace("_", " ")


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not RISK_FILE.exists():
        return pd.DataFrame(), {"error": f"Processed risk file not found: {RISK_FILE}"}
    try:
        raw = pd.read_csv(RISK_FILE)
    except Exception as exc:
        return pd.DataFrame(), {"error": f"Risk file could not be read: {exc}"}
    if raw.empty:
        return pd.DataFrame(), {"error": "Processed risk file is empty."}

    # Canonical internal fields. Original source aliases are accepted only here.
    data = pd.DataFrame(index=raw.index)
    data["cell"] = first_column(raw, ["grid_id", "cell_id", "id"]).astype(str)
    data["district"] = first_column(raw, ["district", "District"]).astype(str).str.strip()
    data["lat"] = numeric(first_column(raw, ["lat", "latitude", "Latitude"]))
    data["lon"] = numeric(first_column(raw, ["lon", "longitude", "Longitude"]))
    data["risk"] = numeric(first_column(raw, ["hyperlocal_risk_score", "risk_score", "risk"]))
    data["susceptibility"] = numeric(first_column(raw, ["susceptibility_score", "spatial_susceptibility", "susceptibility"]))
    data["rain_24h"] = numeric(first_column(raw, ["rainfall_24h", "rain_24h", "rainfall24h"]))
    data["rain_3d"] = numeric(first_column(raw, ["rainfall_3day", "rainfall_3d", "rainfall_72h"]))
    data["rain_7d"] = numeric(first_column(raw, ["rainfall_7day", "rainfall_7d"]))
    data["vulnerability"] = numeric(first_column(raw, ["vulnerability_score", "vulnerability"]))
    data["temporal"] = numeric(first_column(raw, ["temporal_risk_index", "temporal_risk_score", "temporal_signal"]))
    data["driver"] = first_column(raw, ["dominant_driver", "primary_driver", "driver"], "Combined signals").fillna("Combined signals").astype(str)
    data["explanation"] = first_column(raw, ["risk_explanation", "explanation"], "Multiple signals are contributing.").fillna("Multiple signals are contributing.").astype(str)
    data["category"] = first_column(raw, ["hyperlocal_risk_category", "risk_category", "category"]).astype(str)
    data["priority"] = first_column(raw, ["action_priority", "priority"]).astype(str)
    data["date"] = pd.to_datetime(first_column(raw, ["date", "scenario_date"]), errors="coerce")

    if data["risk"].max() <= 1.5:
        data["risk"] = data["risk"] * 100
    if data["temporal"].max() <= 1.5:
        data["temporal"] = data["temporal"] * 100
    data["category"] = data["category"].where(~data["category"].isin(["", "nan", "None"]), data["risk"].map(classify))
    data["priority"] = data["priority"].where(~data["priority"].isin(["", "nan", "None"]), data["risk"].map(priority_for))
    data["district"] = data["district"].replace({"nan": "Unknown district", "": "Unknown district"})
    data = data.replace([np.inf, -np.inf], np.nan)
    data["risk"] = data["risk"].fillna(data["risk"].median()).clip(0, 100)
    for col in ["susceptibility", "rain_24h", "rain_3d", "rain_7d", "vulnerability", "temporal"]:
        data[col] = data[col].fillna(0)
    data = data[data["lat"].between(-90, 90) & data["lon"].between(-180, 180)].copy()
    meta = {
        "rows": len(data),
        "source_columns": list(raw.columns),
        "date_min": data["date"].min(),
        "date_max": data["date"].max(),
        "raw_rows": len(raw),
        "districts": data["district"].nunique(),
        "scenario": data["date"].dropna().iloc[0].strftime("%d %b %Y") if data["date"].notna().any() else "processed project scenario",
    }
    return data.reset_index(drop=True), meta


@st.cache_data(show_spinner=False)
def load_boundaries() -> dict[str, Any] | None:
    if not BOUNDARY_FILE.exists():
        return None
    try:
        return json.loads(BOUNDARY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame()


BIHAR_AFFECTED_2025 = {
    "Patna": 38761,
    "Lakhisarai": 16316,
    "Begusarai": 14484,
    "Bhagalpur": 12205,
    "Nalanda": 11902,
    "Vaishali": 10177,
    "Munger": 7350,
    "Samastipur": 6341,
    "Khagaria": 5982,
    "Saran": 5113,
    "Sheikhpura": 3540,
    "Madhepura": 1211,
    "Purnia": 650,
    "Katihar": 516,
    "Darbhanga": 233,
    "Saharsa": 185,
}

BIHAR_DISTRICTS = [
    "Araria", "Arwal", "Aurangabad", "Banka", "Begusarai", "Bhagalpur",
    "Bhojpur", "Buxar", "Darbhanga", "East Champaran", "Gaya", "Gopalganj",
    "Jamui", "Jehanabad", "Kaimur", "Katihar", "Khagaria", "Kishanganj",
    "Lakhisarai", "Madhepura", "Madhubani", "Munger", "Muzaffarpur",
    "Nalanda", "Nawada", "Patna", "Purnia", "Rohtas", "Saharsa",
    "Samastipur", "Saran", "Sheikhpura", "Sheohar", "Sitamarhi", "Siwan",
    "Supaul", "Vaishali", "West Champaran",
]

BIHAR_NAME_ALIASES = {
    "eastchamparan": ["eastchamparan", "purvichamparan", "purbichamparan"],
    "westchamparan": ["westchamparan", "pashchimchamparan", "paschimchamparan"],
    "kaimur": ["kaimur", "kaimurbhabhua"],
}


def clean_district_name(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def bihar_key(value: Any) -> str:
    key = clean_district_name(value)
    for canonical, aliases in BIHAR_NAME_ALIASES.items():
        if key in aliases:
            return canonical
    return key


@st.cache_data(show_spinner=False)
def load_bihar_validation() -> dict[str, Any]:
    """Build a reproducible district-level Bihar holdout benchmark.

    Training uses only pre-2025 flood inventory history plus static district
    context. The 2025 labels come from the official NRSC satellite benchmark
    embedded above, so the holdout is never used as a training feature.
    """
    required = [BIHAR_FLOODED_AREA_FILE, BIHAR_IMPACT_FILE, BIHAR_INVENTORY_FILE]
    if not all(path.exists() for path in required):
        return {"error": "Bihar evidence data is not available yet."}
    if HistGradientBoostingClassifier is None:
        return {"error": "The analysis tools needed for Bihar evidence are not available yet."}
    try:
        area = pd.read_csv(BIHAR_FLOODED_AREA_FILE)
        impact = pd.read_csv(BIHAR_IMPACT_FILE)
        inventory = pd.read_csv(BIHAR_INVENTORY_FILE, low_memory=False)
    except Exception as exc:
        return {"error": "The Bihar evidence could not be loaded. Please check the project data and reload."}

    area["key"] = area["Dist_Name"].map(bihar_key)
    impact["key"] = impact["Dist_Name"].map(bihar_key)
    observed = pd.DataFrame({"district": BIHAR_DISTRICTS})
    observed["affected_hectares"] = observed["district"].map(BIHAR_AFFECTED_2025).fillna(0)
    observed["key"] = observed["district"].map(bihar_key)
    static = observed[["district", "key", "affected_hectares"]].merge(
        area[["key", "Corrected_Percent_Flooded_Area"]], on="key", how="left"
    ).merge(
        impact[["key", "Population", "Mean_Flood_Duration"]], on="key", how="left"
    )
    static["Corrected_Percent_Flooded_Area"] = numeric(static["Corrected_Percent_Flooded_Area"])
    static["Population"] = numeric(static["Population"])
    static["Mean_Flood_Duration"] = numeric(static["Mean_Flood_Duration"])

    district_keys = static["key"].tolist()
    inventory["year"] = pd.to_datetime(inventory["Start Date"], errors="coerce").dt.year
    inventory = inventory[inventory["year"].between(2015, 2023)].copy()
    inventory["state_text"] = inventory.get("State", "").fillna("").astype(str)
    inventory["district_text"] = inventory.get("Districts", "").fillna("").astype(str)
    events: list[dict[str, Any]] = []
    for row in inventory.itertuples(index=False):
        text = clean_district_name(f"{row.state_text} {row.district_text}")
        for key in district_keys:
            aliases = BIHAR_NAME_ALIASES.get(key, [key])
            if any(alias in text for alias in aliases):
                events.append({"key": key, "year": int(row.year)})
    events_df = pd.DataFrame(events)
    if events_df.empty:
        return {"error": "The available Bihar flood history could not be matched to the district list."}
    counts = events_df.groupby(["key", "year"]).size().rename("flood_events").reset_index()
    years = pd.DataFrame([(key, year) for key in district_keys for year in range(2015, 2024)], columns=["key", "year"])
    panel = years.merge(counts, on=["key", "year"], how="left").fillna({"flood_events": 0})
    panel = panel.sort_values(["key", "year"])
    panel["history_to_date"] = panel.groupby("key")["flood_events"].cumsum() - panel["flood_events"]
    panel = panel.merge(static.drop(columns=["district", "affected_hectares"]), on="key", how="left")
    panel["flood_event"] = (panel["flood_events"] > 0).astype(int)
    features = ["Corrected_Percent_Flooded_Area", "Population", "Mean_Flood_Duration", "history_to_date"]
    train = panel.dropna(subset=features).copy()
    if train["flood_event"].nunique() < 2:
        return {"error": "The available Bihar history does not contain enough contrasting examples for a fair check."}
    model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_leaf_nodes=8, min_samples_leaf=8, random_state=7)
    model.fit(train[features], train["flood_event"])
    latest = static.copy()
    latest["history_to_date"] = latest["key"].map(events_df.groupby("key").size()).fillna(0)
    latest[features] = latest[features].fillna(train[features].median())
    latest["predicted_probability"] = model.predict_proba(latest[features])[:, 1]
    latest["history_score"] = latest["history_to_date"].rank(pct=True)
    latest["static_score"] = latest["Corrected_Percent_Flooded_Area"].rank(pct=True)
    latest["observed_flood"] = (latest["affected_hectares"] > 0).astype(int)
    latest["predicted_flood"] = (latest["predicted_probability"] >= 0.5).astype(int)
    latest["model_rank"] = latest["predicted_probability"].rank(method="first", ascending=False).astype(int)
    latest["history_rank"] = latest["history_score"].rank(method="first", ascending=False).astype(int)
    latest["static_rank"] = latest["static_score"].rank(method="first", ascending=False).astype(int)
    y_true, y_pred = latest["observed_flood"], latest["predicted_flood"]
    positives = int(y_true.sum())
    def ranking_metrics(score_col: str, k: int) -> dict[str, float]:
        ranked = latest.nlargest(k, score_col)
        captured = int(ranked["observed_flood"].sum())
        precision = captured / k if k else 0.0
        recall = captured / positives if positives else 0.0
        random_recall = k / len(latest) if len(latest) else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "lift": recall / random_recall if random_recall else 0.0,
        }
    comparison = []
    for method, score_col in [("FloodSense transfer model", "predicted_probability"), ("History-only baseline", "history_score"), ("Static-area baseline", "static_score")]:
        for k in [5, 10, 16]:
            values = ranking_metrics(score_col, k)
            comparison.append({"Method": method, "Top k": k, "Precision": values["precision"], "Recall": values["recall"], "Lift": values["lift"]})
    confusion = {
        "true_positive": int(((y_true == 1) & (y_pred == 1)).sum()),
        "false_positive": int(((y_true == 0) & (y_pred == 1)).sum()),
        "true_negative": int(((y_true == 0) & (y_pred == 0)).sum()),
        "false_negative": int(((y_true == 1) & (y_pred == 0)).sum()),
    }
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "top16_precision": ranking_metrics("predicted_probability", 16)["precision"],
        "top16_recall": ranking_metrics("predicted_probability", 16)["recall"],
        "top16_lift": ranking_metrics("predicted_probability", 16)["lift"],
        "area_rank_correlation": float(np.nan_to_num(latest["predicted_probability"].corr(latest["affected_hectares"], method="spearman"), nan=0.0)),
    }
    latest["top_rank"] = latest["model_rank"]
    return {
        "table": latest.sort_values("predicted_probability", ascending=False),
        "metrics": metrics,
        "comparison": pd.DataFrame(comparison),
        "confusion": confusion,
        "train_rows": len(train),
        "event_rows": len(events_df),
        "district_count": len(latest),
        "observed_district_count": positives,
        "observed_hectares": int(observed["affected_hectares"].sum()),
        "holdout_year": 2025,
    }


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --ink:#e7f6f8; --muted:#8da8af; --line:#1b3a43; --panel:#0b1b22; --cyan:#55e6e9; }
        .stApp { background: radial-gradient(circle at 70% -10%, #124a55 0, #061116 36%, #040b0e 100%); color:var(--ink); }
        [data-testid="stSidebar"] { background:#061217; border-right:1px solid var(--line); }
        [data-testid="stSidebar"] .block-container { padding:1.2rem .9rem; }
        .block-container { max-width:1500px; padding:1.4rem 2rem 3rem; }
        section.main > div { overflow: visible; }
        h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.03em; }
        p,div,span,label,button { font-family:'DM Sans',sans-serif; }
        .brand { color:var(--cyan); font:700 12px 'Space Grotesk'; letter-spacing:.22em; text-transform:uppercase; }
        .hero { padding:1.3rem 1.5rem; border:1px solid #23525b; border-radius:20px; background:linear-gradient(120deg,rgba(13,47,54,.94),rgba(7,20,25,.92)); margin-bottom:1rem; }
        .hero h1 { margin:.35rem 0 .15rem; font-size:2.1rem; }
        .subtle { color:var(--muted); font-size:.9rem; }
        .panel { background:rgba(9,25,31,.87); border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; height:100%; }
        .panel-title { font:600 1rem 'Space Grotesk'; color:#e5fbfb; }
        .panel-note { color:var(--muted); font-size:.78rem; margin-top:.25rem; }
        .risk-badge { display:inline-block; padding:.25rem .5rem; border-radius:999px; font-size:.72rem; font-weight:700; color:#071114; }
        .why-row { padding:.7rem 0; border-bottom:1px solid #17333b; }
        .why-row:last-child { border:0; }
        .why-label { color:#cdebed; font-weight:600; }
        .why-value { color:#82dadd; float:right; }
        .smallcaps { color:#6e929a; font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700; }
        div[data-testid="stMetric"] { background:#0a1d24; border:1px solid var(--line); border-radius:14px; padding:.7rem; }
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background:#081920; border-color:#23505a; }
        .stButton > button, .stDownloadButton > button { border:1px solid #2b6971; background:#0c2a32; color:#ddffff; border-radius:10px; }
        .stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--cyan); color:white; }
        [data-testid="stSidebar"] .stRadio label { padding:.45rem .45rem; border-radius:9px; }
        [data-testid="stSidebar"] .stRadio label:hover { background:#0d2830; }
        .footnote { color:#78959c; font-size:.74rem; line-height:1.5; }
        .site-footer { margin-top:2.2rem; padding:1.2rem 1.35rem 1rem; border:1px solid #1b3a43; border-radius:16px; background:linear-gradient(120deg,rgba(8,29,36,.96),rgba(5,15,19,.96)); color:#9ab5ba; overflow:hidden; }
        .site-footer-grid { display:grid; grid-template-columns:minmax(0,1.6fr) minmax(0,1fr) minmax(0,1fr) minmax(0,1.2fr); gap:1.25rem; align-items:start; }
        .site-footer-title { color:#e5fbfb; font:600 .95rem 'Space Grotesk'; margin-bottom:.35rem; }
        .site-footer-copy { color:#78959c; font-size:.76rem; line-height:1.5; }
        .site-footer-item { color:#b7d5d8; font-size:.78rem; line-height:1.7; overflow-wrap:anywhere; }
        .site-footer-divider { border:0; border-top:1px solid #18353d; margin:1rem 0 .75rem; }
        .site-footer-bottom { display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; color:#6e8d94; font-size:.7rem; }
        @media (max-width: 900px) { .site-footer-grid { grid-template-columns:1fr 1fr; } }
        @media (max-width: 560px) { .site-footer-grid { grid-template-columns:1fr; } .site-footer-bottom { display:block; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def go_to(page: str) -> None:
    st.session_state["page"] = page


def navigation() -> None:
    st.sidebar.markdown('<div class="brand">FLOODSENSE AI</div>', unsafe_allow_html=True)
    st.sidebar.caption("Flood safety planning · NCR")
    options = [f"{PAGE_ICONS[p]}  {PAGE_LABELS[p]}" for p in PAGES]
    current = st.session_state.get("page", PAGES[0])
    if current not in PAGES:
        current = PAGES[0]
    current_label = f"{PAGE_ICONS[current]}  {PAGE_LABELS[current]}"
    selected = st.sidebar.radio("Navigate", options, index=options.index(current_label), label_visibility="collapsed")
    selected_page = next((page for page in PAGES if selected == f"{PAGE_ICONS[page]}  {PAGE_LABELS[page]}"), PAGES[0])
    st.session_state["page"] = selected_page
    st.sidebar.divider()
    st.sidebar.markdown('<div class="smallcaps">Session</div>', unsafe_allow_html=True)
    st.sidebar.caption("Reference view: project scenario")
    if st.sidebar.button("Reset filters", use_container_width=True):
        for key in ["district_filter", "cell_choice", "cell_label_choice", "why_label_choice", "scenario_preset", "rain_multiplier"]:
            st.session_state.pop(key, None)
        st.rerun()


def page_frame(title: str, subtitle: str, data: pd.DataFrame) -> None:
    st.markdown(f'<div class="hero"><div class="brand">FLOODSENSE AI · NCR</div><h1>{html.escape(title)}</h1><div class="subtle">{html.escape(subtitle)}</div></div>', unsafe_allow_html=True)
    if data.empty:
        st.error("The flood-risk information is not available yet. Please check the project setup and reload.")


def nav_buttons(page: str) -> None:
    idx = PAGES.index(page)
    a, b, c = st.columns([1, 1, 1])
    with a:
        if idx > 0 and st.button("← Back", use_container_width=True, key=f"back_{page}"):
            go_to(PAGES[idx - 1]); st.rerun()
    with b:
        if st.button("⌂ Command Center", use_container_width=True, key=f"home_{page}"):
            go_to(PAGES[0]); st.rerun()
    with c:
        if idx < len(PAGES) - 1 and st.button("Next →", use_container_width=True, key=f"next_{page}"):
            go_to(PAGES[idx + 1]); st.rerun()


def render_footer(meta: dict[str, Any]) -> None:
    rows = int(meta.get("rows", 0) or 0)
    districts = meta.get("districts", "—")
    scenario = meta.get("scenario", "reference view")
    st.markdown(
        f"""
        <footer class="site-footer">
          <div class="site-footer-grid">
            <div>
              <div class="brand">FLOODSENSE AI · NCR</div>
              <div class="site-footer-title">Flood intelligence for better first decisions.</div>
              <div class="site-footer-copy">A clear demonstration interface for exploring location-level flood concern, understanding why it appears, and planning attention.</div>
            </div>
            <div>
              <div class="smallcaps">Coverage</div>
              <div class="site-footer-item">{rows:,} locations on the map</div>
              <div class="site-footer-item">{html.escape(str(districts))} districts</div>
            </div>
            <div>
              <div class="smallcaps">Explore</div>
              <div class="site-footer-item">Flood map · Area explanation</div>
              <div class="site-footer-item">What-if rainfall · District comparison</div>
            </div>
            <div>
              <div class="smallcaps">Status</div>
              <div class="site-footer-item">● Reference view loaded</div>
              <div class="site-footer-item">○ What-if views clearly marked</div>
            </div>
          </div>
          <hr class="site-footer-divider">
          <div class="site-footer-bottom">
            <span>Reference date: {html.escape(str(scenario))}</span>
            <span>Planning aid only · Not a live warning or forecast</span>
          </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(label: str) -> str:
    color = RISK_COLORS.get(label, "#9ca3af")
    return f'<span class="risk-badge" style="background:{color}">{html.escape(label)}</span>'


def score_chart(df: pd.DataFrame, title: str = "Flood concern distribution") -> None:
    if px is None:
        st.dataframe(df.assign(District=df["district"], Concern=df["risk"].round(1))[["District", "Concern"]].head(100), use_container_width=True, hide_index=True)
        return
    fig = px.histogram(df, x="risk", nbins=24, color="category", color_discrete_map=RISK_COLORS, template="plotly_dark")
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=25, b=0), title=title, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


@st.cache_data(show_spinner=False)
def map_points(df: pd.DataFrame, limit: int = MAP_POINT_LIMIT) -> pd.DataFrame:
    """Prepare a stable, lightweight map layer for fast reruns.

    The full 8,471-cell dataset remains available for tables and calculations.
    Only the visual layer is reduced: the highest-risk cells are retained and
    the remainder is sampled deterministically so the map still shows the NCR
    field rather than only a small hotspot.
    """
    if len(df) <= limit:
        return df.copy()
    hotspot_count = min(350, limit // 2)
    hotspots = df.nlargest(hotspot_count, "risk")
    remaining = df.drop(hotspots.index)
    sample_count = limit - len(hotspots)
    sample = remaining.sample(n=min(sample_count, len(remaining)), random_state=42)
    return pd.concat([hotspots, sample]).drop_duplicates("cell")


def build_map(df: pd.DataFrame, selected_cell: str | None = None, layer: str = "Risk"):
    if folium is None or df.empty:
        return None
    center = [float(df["lat"].mean()), float(df["lon"].mean())]
    fmap = folium.Map(location=center, zoom_start=9, tiles="OpenStreetMap", control_scale=True)
    Fullscreen(position="topright").add_to(fmap)
    boundaries = load_boundaries()
    if boundaries:
        folium.GeoJson(boundaries, name="District boundaries", style_function=lambda _: {"color": "#56bfc5", "weight": 1, "fillOpacity": 0}).add_to(fmap)
    points = map_points(df)
    cluster = MarkerCluster(name="Map locations", showCoverageOnHover=False).add_to(fmap)
    for row in points.itertuples(index=False):
        if layer == "Risk":
            label = row.category
        elif layer == "Rainfall":
            label = f"{row.rain_24h:.1f} mm / 24h"
        elif layer == "Susceptibility":
            label = f"{row.susceptibility:.1f} / 100"
        else:
            label = f"{row.vulnerability:.1f} / 100"
        color = RISK_COLORS.get(row.category, "#38bdf8") if layer == "Risk" else "#38bdf8"
        selected = row.cell == selected_cell
        popup = f"<b>{html.escape(friendly_location(row.cell))}</b><br>{html.escape(row.district)}<br>{html.escape(label)}<br><br>Use the area selector beside the map for the full explanation."
        folium.CircleMarker([row.lat, row.lon], radius=7 if selected else 4, color="#ffffff" if selected else color, weight=2 if selected else 1, fill=True, fill_color=color, fill_opacity=.9, popup=folium.Popup(popup, max_width=250), tooltip=f"{html.escape(row.district)} · {row.risk:.1f}").add_to(cluster)
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


def selected_row(data: pd.DataFrame) -> pd.Series:
    choices = data["cell"].tolist()
    labels = [friendly_location(choice) for choice in choices]
    prior_label = st.session_state.get("cell_label_choice", labels[0] if labels else "")
    if prior_label not in labels:
        prior_label = labels[0] if labels else ""
    selected_label = st.selectbox("Choose an area", labels, index=labels.index(prior_label) if prior_label in labels else 0, key="cell_label_choice") if choices else ""
    choice = choices[labels.index(selected_label)] if selected_label in labels else ""
    if choice:
        return data[data["cell"] == choice].iloc[0]
    return pd.Series(dtype=object)


def why_panel(row: pd.Series) -> None:
    if row.empty:
        st.info("Choose an area to see its explanation.")
        return
    st.markdown(f'<div class="panel"><div class="smallcaps">Area selected</div><div class="panel-title">{html.escape(friendly_location(row.cell))} · {html.escape(str(row.district))}</div><div style="margin:.65rem 0">{risk_badge(str(row.category))} <span class="subtle">&nbsp; {float(row.risk):.1f} / 100 concern</span></div><div class="why-row"><span class="why-label">Main reason</span><span class="why-value">{html.escape(friendly_driver(row.driver))}</span></div><div class="why-row"><span class="why-label">Rain in the last day</span><span class="why-value">{float(row.rain_24h):.1f} mm</span></div><div class="why-row"><span class="why-label">Ground and surface conditions</span><span class="why-value">{float(row.susceptibility):.1f} / 100</span></div><div class="why-row"><span class="why-label">People and property exposure</span><span class="why-value">{float(row.vulnerability):.1f} / 100</span></div><p class="footnote">{html.escape(friendly_explanation(row.explanation))}</p></div>', unsafe_allow_html=True)


def page_command_center(data: pd.DataFrame) -> None:
    page_frame("Where should we look first?", "A practical operating view for locating, explaining, and acting on urban flood risk.", data)
    if data.empty:
        return
    mean_risk = data["risk"].mean()
    top = data.nlargest(1, "risk").iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Locations reviewed", f"{len(data):,}", "NCR map")
    c2.metric("Average flood concern", f"{mean_risk:.1f}/100", "reference view")
    c3.metric("Locations to check first", f"{data['priority'].isin(['High Priority','Immediate Review']).sum():,}", "highest concern")
    c4.metric("Districts included", f"{data['district'].nunique()}", "project data")
    st.markdown("### Start with a question")
    q = st.columns(4)
    actions = [("Open flood map", "Risk Map"), ("Explain one area", "Why This Area?"), ("See what to check first", "Intervention Priority"), ("Try a rainfall what-if", "Scenario Lab")]
    for col, (label, target) in zip(q, actions):
        with col:
            if st.button(label, use_container_width=True):
                go_to(target); st.rerun()
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Flood concern map</div><div class="panel-note">Areas needing the most attention in the reference view.</div>', unsafe_allow_html=True)
        m = build_map(data, str(top["cell"]))
        if m is not None:
            st_folium(m, height=470, use_container_width=True, returned_objects=[])
        else:
            st.dataframe(data.nlargest(20, "risk").assign(Area=lambda frame: frame["cell"].map(friendly_location), District=lambda frame: frame["district"], Concern=lambda frame: frame["risk"].round(1), Level=lambda frame: frame["category"])[["Area", "District", "Concern", "Level"]], use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Highest-concern area</div><div class="panel-note">Based on the reference score.</div>', unsafe_allow_html=True)
        why_panel(top)
        st.markdown("</div>", unsafe_allow_html=True)
    score_chart(data)
    st.markdown('<p class="footnote">This is a planning simulation built from the project’s reference data. It is not a live warning service or a forecast.</p>', unsafe_allow_html=True)
    nav_buttons("Command Center")


def page_risk_map(data: pd.DataFrame) -> None:
    page_frame("Flood Map", "Explore the NCR map, switch what you want to see, and select an area for details.", data)
    if data.empty:
        return
    c1, c2 = st.columns([1, 1])
    layer_label = c1.selectbox("Show on map", ["Flood concern", "Rainfall", "Ground conditions", "People and property"])
    layer = {"Flood concern": "Risk", "Rainfall": "Rainfall", "Ground conditions": "Susceptibility", "People and property": "Exposure"}[layer_label]
    district = c2.selectbox("Choose district", ["All districts"] + sorted(data["district"].unique().tolist()))
    shown = data if district == "All districts" else data[data["district"] == district]
    mcol, icol = st.columns([1.65, 1])
    with icol:
        row = selected_row(shown)
    with mcol:
        m = build_map(shown, str(row.get("cell", "")), layer)
        if m is not None:
            st_folium(m, height=610, use_container_width=True, returned_objects=[])
        else:
            st.dataframe(shown.assign(Area=shown["cell"].map(friendly_location), District=shown["district"], Latitude=shown["lat"].round(5), Longitude=shown["lon"].round(5), Concern=shown["risk"].round(1), Level=shown["category"])[["Area", "District", "Latitude", "Longitude", "Concern", "Level"]].head(100), use_container_width=True, hide_index=True)
        st.caption("The map uses the project's location data and a public street map. No sign-in is required.")
    with icol:
        why_panel(row)
    nav_buttons("Risk Map")


def page_why(data: pd.DataFrame) -> None:
    page_frame("Why This Area?", "Understand why an area is highlighted and what to check next.", data)
    if data.empty:
        return
    top = data.nlargest(100, "risk")
    choices = top["cell"].tolist()
    labels = [friendly_location(choice) for choice in choices]
    prior_label = st.session_state.get("why_label_choice", labels[0])
    if prior_label not in labels:
        prior_label = labels[0]
    selected_label = st.selectbox("Choose an area", labels, index=labels.index(prior_label), key="why_label_choice")
    choice = choices[labels.index(selected_label)]
    row = data[data["cell"] == choice].iloc[0]
    left, right = st.columns([1, 1.4])
    with left:
        why_panel(row)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">What to do next</div><div class="panel-note">This is a planning aid, not an automatic dispatch order.</div>', unsafe_allow_html=True)
        action = "Inspect drainage and access routes first" if row.driver in ["Slope/runoff", "Built-up land"] else "Check rain gauges, low points, and known flood routes"
        st.markdown(f"### {action}")
        st.write("Use this area to focus a field check, compare nearby areas, and decide whether protective resources should be staged.")
        st.markdown(f"**Suggested attention:** {friendly_priority(row.priority)}")
        st.markdown("</div>", unsafe_allow_html=True)
    nav_buttons("Why This Area?")


def page_priority(data: pd.DataFrame) -> None:
    page_frame("What Should We Check First?", "A clear, explainable list for deciding where limited attention should go first.", data)
    if data.empty:
        return
    district = st.selectbox("Focus on a district", ["All districts"] + sorted(data["district"].unique().tolist()))
    shown = data if district == "All districts" else data[data["district"] == district]
    n = st.slider("Number of areas to show", 5, 30, 12)
    top = shown.sort_values("risk", ascending=False).head(n).copy()
    display = top.assign(Concern=top["risk"].round(1), Attention=top["priority"].map(friendly_priority), District=top["district"], Area=top["cell"].map(friendly_location))[["Area", "District", "Concern", "Attention"]]
    st.dataframe(display, use_container_width=True, hide_index=True)
    priority_download = top.assign(Area=top["cell"].map(friendly_location), District=top["district"], Concern=top["risk"].round(1), Level=top["category"], Attention=top["priority"].map(friendly_priority), Main_reason=top["driver"].map(friendly_driver))[["Area", "District", "Concern", "Level", "Attention", "Main_reason"]]
    st.download_button("Download this check list", priority_download.to_csv(index=False).encode("utf-8"), "floodsense_check_list.csv", "text/csv")
    st.markdown("### Why these are first")
    cols = st.columns(3)
    for col, row in zip(cols, top.head(3).itertuples(index=False)):
        with col:
            st.markdown(f'<div class="panel"><div class="smallcaps">{html.escape(friendly_priority(row.priority))}</div><div class="panel-title">{html.escape(friendly_location(row.cell))}</div><p class="subtle">{html.escape(row.district)} · {row.risk:.1f}/100 concern</p><p>{html.escape(friendly_driver(row.driver))}</p></div>', unsafe_allow_html=True)
    nav_buttons("Intervention Priority")


def page_planner(data: pd.DataFrame) -> None:
    page_frame("Response Planner", "Turn the check list into a transparent, adjustable plan.", data)
    if data.empty:
        return
    total = st.number_input("Available response teams", min_value=1, max_value=1000, value=25, step=1)
    summary = data.groupby("district", as_index=False).agg(risk=("risk", "mean"), priority_locations=("priority", lambda s: s.isin(["High Priority", "Immediate Review"]).sum()), locations=("cell", "count"))
    summary["weight"] = summary["risk"] * (1 + summary["priority_locations"] / summary["locations"].clip(lower=1))
    summary["units"] = np.floor(summary["weight"] / summary["weight"].sum() * total).astype(int)
    remainder = int(total - summary["units"].sum())
    if remainder > 0:
        summary.loc[summary["weight"].idxmax(), "units"] += remainder
    display = summary.rename(columns={"district": "District", "risk": "Average concern", "priority_locations": "Locations to check first", "units": "Suggested teams"}).drop(columns="weight").round(1)
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("Suggested teams are allocated using observed concern and the number of locations needing attention. Adjust locally before real-world use.")
    nav_buttons("Response Planner")


def page_scenario(data: pd.DataFrame) -> None:
    page_frame("Rainfall What-If", "Change rainfall conditions and see the displayed flood concern recalculate. Every result here is clearly marked as simulated.", data)
    if data.empty:
        return
    left, right = st.columns([1, 1.8])
    with left:
        preset = st.selectbox("Rainfall situation", ["Base conditions", "Drier conditions", "Heavy rainfall", "Extreme rainfall"])
        multipliers = {"Base conditions": 1.0, "Drier conditions": .65, "Heavy rainfall": 1.35, "Extreme rainfall": 1.8}
        multiplier = multipliers[preset]
        custom = st.slider("Rainfall change", .4, 2.2, float(multiplier), .05)
        st.info("What-if view only — it does not change the reference data or represent a real forecast.")
    sim = data.copy()
    rain_pressure = minmax(sim["rain_24h"] * custom + sim["rain_7d"] * custom / 7)
    sim["sim_risk"] = (sim["risk"] * .72 + rain_pressure * .28).clip(0, 100)
    sim["sim_category"] = sim["sim_risk"].map(classify)
    with right:
        avg_before, avg_after = float(sim["risk"].mean()), float(sim["sim_risk"].mean())
        a, b, c = st.columns(3)
        a.metric("Reference concern", f"{avg_before:.1f}/100")
        b.metric("What-if concern", f"{avg_after:.1f}/100", f"{avg_after-avg_before:+.1f}")
        c.metric("Locations needing attention", f"{(sim['sim_risk'] >= 40).sum():,}", f"reference {(sim['risk'] >= 40).sum():,}")
        if px is not None:
            chart = sim.sample(min(SCENARIO_POINT_LIMIT, len(sim)), random_state=7)
            fig = px.scatter(chart, x="risk", y="sim_risk", color="sim_category", color_discrete_map=RISK_COLORS, template="plotly_dark", labels={"risk": "Reference concern", "sim_risk": "What-if concern"})
            fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="#78909c", dash="dot"))
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=25, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("### Locations most affected by this what-if")
    affected = sim.assign(Change=(sim["sim_risk"] - sim["risk"]).round(1), Simulated_Risk=sim["sim_risk"].round(1)).sort_values("Change", ascending=False)
    affected_display = affected.assign(Area=affected["cell"].map(friendly_location), District=affected["district"], **{"Reference concern": affected["risk"].round(1), "What-if concern": affected["Simulated_Risk"], "Change": affected["Change"]})[["Area", "District", "Reference concern", "What-if concern", "Change"]]
    st.dataframe(affected_display.head(15), use_container_width=True, hide_index=True)
    nav_buttons("Scenario Lab")


def page_compare(data: pd.DataFrame) -> None:
    page_frame("Compare Districts", "See how flood concern differs across districts in the reference view.", data)
    if data.empty:
        return
    summary = data.groupby("district", as_index=False).agg(Cells=("cell", "count"), Average_Risk=("risk", "mean"), Peak_Risk=("risk", "max"), Rain_24h=("rain_24h", "mean"), Exposure=("vulnerability", "mean")).sort_values("Average_Risk", ascending=False)
    display = summary.rename(columns={"district": "District", "Average_Risk": "Average concern", "Peak_Risk": "Highest concern", "Rain_24h": "Average rain (24h)", "Exposure": "People/property exposure"}).round(1)
    st.dataframe(display, use_container_width=True, hide_index=True)
    if px is not None:
        fig = px.bar(summary, x="district", y="Average_Risk", color="Average_Risk", color_continuous_scale="Tealgrn", template="plotly_dark", labels={"district": "District", "Average_Risk": "Average concern"})
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    nav_buttons("Compare Districts")


def page_bihar_validation() -> None:
    st.markdown('<div class="hero"><div class="brand">FLOODSENSE AI · BIHAR EVIDENCE</div><h1>How well did the Bihar test perform?</h1><div class="subtle">A fair district-by-district check against independently observed flooding.</div></div>', unsafe_allow_html=True)
    result = load_bihar_validation()
    if result.get("error"):
        st.error(result["error"])
        st.info("This evidence view needs the project's Bihar reference files and supporting analysis tools.")
        nav_buttons("Bihar Validation")
        return
    metrics = result["metrics"]
    st.warning("This Bihar test works at district level, while the NCR map works at neighbourhood-map level. It uses past flood records and long-term district conditions, then checks them against independent 2025 observations.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Correct high-concern picks", f"{metrics['precision']:.0%}")
    c2.metric("Affected districts found", f"{metrics['recall']:.0%}")
    c3.metric("Overall balance", f"{metrics['f1']:.0%}")
    c4.metric("Top-16 coverage", f"{metrics['top16_recall']:.0%}", f"{metrics['top16_lift']:.1f}× random")
    st.info(f"Scope: {result['observed_district_count']} of {result['district_count']} Bihar districts were independently observed as affected in {result['holdout_year']}, covering {result['observed_hectares']:,} hectares. This is a district-ranking check, not a neighbourhood-level Bihar forecast.")
    st.caption(f"Past records used: {result['train_rows']:,} district-by-year entries · matched flood reports: {result['event_rows']:,} · First 16 correct picks: {metrics['top16_precision']:.0%} · Balanced consistency: {metrics['balanced_accuracy']:.0%} · Match with flooded area: {metrics['area_rank_correlation']:.0%}")
    st.markdown("### Why this is stronger evidence")
    e1, e2, e3 = st.columns(3)
    e1.markdown('<div class="panel"><div class="panel-title">Independent check</div><p class="site-footer-copy">The 2025 satellite observations are kept separate from the information used to create the ranking.</p></div>', unsafe_allow_html=True)
    e2.markdown('<div class="panel"><div class="panel-title">Fair comparison</div><p class="site-footer-copy">The full approach is compared with simpler ways of ordering districts.</p></div>', unsafe_allow_html=True)
    e3.markdown('<div class="panel"><div class="panel-title">Useful for planning</div><p class="site-footer-copy">The short-list measure shows how many affected districts appear in a limited inspection queue.</p></div>', unsafe_allow_html=True)
    st.markdown("### Full approach versus simpler approaches")
    comparison = result["comparison"].copy()
    comparison = comparison.rename(columns={"Method": "Approach", "Top k": "Short-list size", "Precision": "Correct picks", "Recall": "Affected districts found", "Lift": "Advantage vs random"})
    comparison["Correct picks"] = comparison["Correct picks"].map(lambda value: f"{value:.0%}")
    comparison["Affected districts found"] = comparison["Affected districts found"].map(lambda value: f"{value:.0%}")
    comparison["Advantage vs random"] = comparison["Advantage vs random"].map(lambda value: f"{value:.1f}×")
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    confusion = result["confusion"]
    st.caption(f"Using the standard decision cutoff: {confusion['true_positive']} affected districts correctly identified · {confusion['false_negative']} missed · {confusion['false_positive']} false alarms · {confusion['true_negative']} correctly left off the list.")
    st.markdown("### What the test is checking")
    st.write("The ranking uses information available before the 2025 event. It is compared with the officially observed affected districts; the hectares below are observations, not predictions.")
    table = result["table"].copy()
    table["Flood concern likelihood"] = (table["predicted_probability"] * 100).round(1)
    table["Observed flooded area (ha)"] = table["affected_hectares"].round(0).astype(int)
    table["Observed flooding"] = np.where(table["observed_flood"].eq(1), "Yes", "No")
    table["Order"] = table["model_rank"].astype(int)
    display = table.sort_values("Order")[["Order", "district", "Flood concern likelihood", "Observed flooded area (ha)", "Observed flooding"]].rename(columns={"district": "District"})
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button("Download Bihar evidence table", display.to_csv(index=False).encode("utf-8"), "floodsense_bihar_evidence_2025.csv", "text/csv")
    if px is not None:
        chart = result["table"].copy().sort_values("predicted_probability", ascending=True)
        chart["Observed flooding"] = np.where(chart["observed_flood"].eq(1), "Observed", "Not observed")
        fig = px.bar(chart, x="predicted_probability", y="district", orientation="h", color="Observed flooding", color_discrete_map={"Observed": "#55e6e9", "Not observed": "#31545d"}, template="plotly_dark", labels={"predicted_probability": "Flood concern likelihood", "district": "District"})
        fig.update_layout(height=650, margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("### Evidence used")
    st.markdown("- [NRSC / ISRO Bihar inundation report, 29 August 2025](https://ndem.nrsc.gov.in/documents/Disaster_Document/2025/BR/brflood50dsc29082025_0600hrs/brflood50dsc29082025_0600hrs_report.pdf): 134,966 hectares across 16 districts.\n- [PIB Bihar flood assessment, 16 September 2026](https://www.pib.gov.in/PressReleseDetailm.aspx?PRID=2311105&lang=2&reg=48): separate recent event context, not used as a 2025 label.")
    st.markdown("### Important interpretation")
    st.write("A strong result here would support district-level ranking transferability. It would not prove that the NCR location-level model is accurate in Bihar. A true Bihar fine-map check still requires Bihar-wide terrain, land-cover, rainfall and inundation layers.")
    nav_buttons("Bihar Validation")


def page_methodology(data: pd.DataFrame, meta: dict[str, Any], history: pd.DataFrame) -> None:
    page_frame("Methodology & Model Insights", "Technical details are kept here so the operating pages stay readable and action-oriented.", data)
    st.markdown("### What the product does")
    st.write("FloodSense combines a spatial susceptibility surface, rainfall-linked temporal signal, and vulnerability context into a hyperlocal risk score for the NCR grid. The result is a prioritization aid for exploration and response planning.")
    st.markdown("### Model and data facts")
    facts = st.columns(4)
    facts[0].metric("Source rows", f"{meta.get('raw_rows', 0):,}")
    facts[1].metric("Usable cells", f"{meta.get('rows', 0):,}")
    facts[2].metric("Districts", f"{data['district'].nunique() if not data.empty else 0}")
    facts[3].metric("Historical context", "Available" if not history.empty else "Not loaded")
    with st.expander("Internal field mapping and validation", expanded=False):
        st.write("The loader accepts the project's original names and maps them once to stable internal fields. This prevents map rendering and charts from depending on renamed display labels.")
        st.code("\n".join(meta.get("source_columns", [])) or "No source columns found")
        st.write("Validated signals: location, district, risk score, rainfall windows, susceptibility, vulnerability, driver, explanation, category, and priority.")
    with st.expander("Scientific caveats", expanded=True):
        st.write("The displayed base state is a historical project scenario, not live sensor data. The Scenario Lab applies a transparent simulated rainfall adjustment for demonstration; it does not retrain the model, update the source file, or produce a real forecast. Local field verification and official warnings remain authoritative.")
    if not data.empty:
        st.markdown("### Available source coverage")
        st.dataframe(data[["district", "risk", "rain_24h", "rain_3d", "rain_7d", "susceptibility", "vulnerability", "temporal"]].describe().round(2), use_container_width=True)
    nav_buttons("Methodology & Model Insights")


def main() -> None:
    st.set_page_config(page_title="FloodSense AI · NCR", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    data, meta = load_data()
    history = load_history()
    navigation()
    page = st.session_state.get("page", PAGES[0])
    pages = {
        "Command Center": lambda: page_command_center(data),
        "Risk Map": lambda: page_risk_map(data),
        "Why This Area?": lambda: page_why(data),
        "Intervention Priority": lambda: page_priority(data),
        "Response Planner": lambda: page_planner(data),
        "Scenario Lab": lambda: page_scenario(data),
        "Compare Districts": lambda: page_compare(data),
        "Bihar Validation": page_bihar_validation,
        "Methodology & Model Insights": lambda: page_methodology(data, meta, history),
    }
    pages.get(page, pages["Command Center"])()
    render_footer(meta)


if __name__ == "__main__":
    main()
