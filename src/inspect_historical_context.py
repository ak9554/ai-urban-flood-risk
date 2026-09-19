import pandas as pd
from pathlib import Path


FLOOD_AREA_FILE = Path(
    "data/raw/District_FloodedArea.csv"
)

FLOOD_IMPACT_FILE = Path(
    "data/raw/District_FloodImpact.csv"
)

FLOOD_LABEL_FILE = Path(
    "data/processed/ncr_flood_labels_2015_2023.csv"
)


print("=" * 60)
print("Inspecting historical flood context")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load district flood area
# ------------------------------------------------------------

area = pd.read_csv(
    FLOOD_AREA_FILE
)

impact = pd.read_csv(
    FLOOD_IMPACT_FILE
)

labels = pd.read_csv(
    FLOOD_LABEL_FILE
)


# ------------------------------------------------------------
# 2. Normalize names
# ------------------------------------------------------------

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
# 3. Seven NCR districts
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
# 4. Filter
# ------------------------------------------------------------

area_ncr = area[
    area["district"].isin(
        ncr_districts
    )
].copy()

impact_ncr = impact[
    impact["district"].isin(
        ncr_districts
    )
].copy()


# ------------------------------------------------------------
# 5. Flood-event frequency
# ------------------------------------------------------------

event_counts = (
    labels
    .groupby("district")
    .size()
    .reset_index(
        name="flood_label_days"
    )
)


# ------------------------------------------------------------
# 6. Combine historical context
# ------------------------------------------------------------

context = area_ncr[
    [
        "district",
        "Percent_Flooded_Area",
        "Parmanent_Water",
        "Corrected_Percent_Flooded_Area"
    ]
].merge(
    impact_ncr[
        [
            "district",
            "Human_fatality",
            "Human_injured",
            "Population",
            "Mean_Flood_Duration"
        ]
    ],
    on="district",
    how="outer"
).merge(
    event_counts,
    on="district",
    how="left"
)


# ------------------------------------------------------------
# 7. Fill event count
# ------------------------------------------------------------

context["flood_label_days"] = (
    context["flood_label_days"]
    .fillna(0)
    .astype(int)
)


# ------------------------------------------------------------
# 8. Display
# ------------------------------------------------------------

print()
print("NCR districts found:", len(context))

print()
print(
    context.to_string(index=False)
)


# ------------------------------------------------------------
# 9. Missing values
# ------------------------------------------------------------

print()
print("Missing values:")

print(
    context.isna()
    .sum()
    .to_string()
)


print()
print("=" * 60)
print("Historical context inspection complete.")
print("=" * 60)