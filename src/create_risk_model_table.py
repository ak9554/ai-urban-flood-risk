import pandas as pd
from pathlib import Path


TRAINING_FILE = Path(
    "data/processed/ncr_district_day_training.csv"
)

CONTEXT_FILE = Path(
    "data/raw/District_FloodedArea.csv"
)

IMPACT_FILE = Path(
    "data/raw/District_FloodImpact.csv"
)

FLOOD_LABEL_FILE = Path(
    "data/processed/ncr_flood_labels_2015_2023.csv"
)

OUTPUT_FILE = Path(
    "data/processed/ncr_risk_model_training.csv"
)


print("=" * 60)
print("Creating NCR district-level flood risk model table")
print("=" * 60)


# ------------------------------------------------------------
# 1. NCR study districts
# ------------------------------------------------------------

ncr_districts = [
    "NEW DELHI",
    "GURUGRAM",
    "FARIDABAD",
    "PANIPAT",
    "SONIPAT",
    "GAUTAM BUDDHA NAGAR",
    "GHAZIABAD"
]


# ------------------------------------------------------------
# 2. Load datasets
# ------------------------------------------------------------

training = pd.read_csv(
    TRAINING_FILE,
    parse_dates=["date"]
)

area = pd.read_csv(
    CONTEXT_FILE
)

impact = pd.read_csv(
    IMPACT_FILE
)

labels = pd.read_csv(
    FLOOD_LABEL_FILE
)


# ------------------------------------------------------------
# 3. Normalize district names
# ------------------------------------------------------------

training["district"] = (
    training["district"]
    .str.strip()
    .str.upper()
)

area["district"] = (
    area["Dist_Name"]
    .str.strip()
    .str.upper()
)

impact["district"] = (
    impact["Dist_Name"]
    .str.strip()
    .str.upper()
)

labels["district"] = (
    labels["district"]
    .str.strip()
    .str.upper()
)


# ------------------------------------------------------------
# 4. Filter ALL datasets to NCR
# ------------------------------------------------------------

training = training[
    training["district"].isin(ncr_districts)
].copy()

area = area[
    area["district"].isin(ncr_districts)
].copy()

impact = impact[
    impact["district"].isin(ncr_districts)
].copy()

labels = labels[
    labels["district"].isin(ncr_districts)
].copy()


# ------------------------------------------------------------
# 5. Calculate historical flood frequency
# ------------------------------------------------------------

flood_days = (
    labels
    .groupby("district")
    .size()
    .reset_index(
        name="historical_flood_days"
    )
)


# ------------------------------------------------------------
# 6. Prepare flooded-area context
# ------------------------------------------------------------

area_context = area[
    [
        "district",
        "Corrected_Percent_Flooded_Area"
    ]
].copy()


area_context = area_context.rename(
    columns={
        "Corrected_Percent_Flooded_Area":
        "historical_flooded_area_percent"
    }
)


# ------------------------------------------------------------
# 7. Prepare population and duration context
# ------------------------------------------------------------

impact_context = impact[
    [
        "district",
        "Population",
        "Mean_Flood_Duration"
    ]
].copy()


impact_context = impact_context.rename(
    columns={
        "Population":
        "population",

        "Mean_Flood_Duration":
        "historical_mean_flood_duration"
    }
)


# ------------------------------------------------------------
# 8. Combine historical context
# ------------------------------------------------------------

context = area_context.merge(
    impact_context,
    on="district",
    how="inner"
)

context = context.merge(
    flood_days,
    on="district",
    how="left"
)


context["historical_flood_days"] = (
    context["historical_flood_days"]
    .fillna(0)
    .astype(int)
)


# ------------------------------------------------------------
# 9. Verify exactly 7 districts
# ------------------------------------------------------------

print()
print("NCR districts in context table:")

print(
    context["district"]
    .sort_values()
    .to_string(index=False)
)

print()
print(
    "Number of NCR districts:",
    context["district"].nunique()
)


# ------------------------------------------------------------
# 10. Attach context to district-day training data
# ------------------------------------------------------------

risk_table = training.merge(
    context,
    on="district",
    how="left"
)


# ------------------------------------------------------------
# 11. Check results
# ------------------------------------------------------------

print()
print("Rows:", len(risk_table))

print()
print("Columns:")

print(
    risk_table.columns.tolist()
)


print()
print("Missing values:")

print(
    risk_table.isna()
    .sum()
    .to_string()
)


print()
print("Flood class distribution:")

print(
    risk_table["flood_event"]
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print("Rows by district:")

print(
    risk_table["district"]
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print("Historical context:")

print(
    context
    .sort_values("district")
    .to_string(index=False)
)


# ------------------------------------------------------------
# 12. Save
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

risk_table.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("Saved:", OUTPUT_FILE)

print()
print("=" * 60)
print("NCR risk model table created successfully.")
print("=" * 60)