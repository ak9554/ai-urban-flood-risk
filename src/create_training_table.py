import pandas as pd
from pathlib import Path


RAINFALL_FILE = Path(
    "data/raw/rainfall/ncr_rainfall_2015_2023.csv"
)

FLOOD_FILE = Path(
    "data/processed/ncr_flood_labels_2015_2023.csv"
)

OUTPUT_FILE = Path(
    "data/processed/ncr_district_day_training.csv"
)


print("=" * 60)
print("Creating district-day historical training table")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load rainfall
# ------------------------------------------------------------

rainfall = pd.read_csv(
    RAINFALL_FILE
)

print()
print("Rainfall rows:", len(rainfall))


# ------------------------------------------------------------
# 2. Load flood labels
# ------------------------------------------------------------

flood = pd.read_csv(
    FLOOD_FILE
)

print(
    "Flood labels:",
    len(flood)
)


# ------------------------------------------------------------
# 3. Normalize district names
# ------------------------------------------------------------

rainfall["district"] = (
    rainfall["district"]
    .str.strip()
    .str.upper()
)

flood["district"] = (
    flood["district"]
    .str.strip()
    .str.upper()
)


# ------------------------------------------------------------
# 4. Convert dates
# ------------------------------------------------------------

rainfall["date"] = pd.to_datetime(
    rainfall["date"]
)

flood["date"] = pd.to_datetime(
    flood["date"]
)


# ------------------------------------------------------------
# 5. Create flood lookup
# ------------------------------------------------------------

flood_lookup = (
    flood[
        [
            "district",
            "date",
            "flood_event"
        ]
    ]
    .drop_duplicates(
        subset=[
            "district",
            "date"
        ]
    )
)


# ------------------------------------------------------------
# 6. Merge flood labels onto rainfall
# ------------------------------------------------------------

training = rainfall.merge(
    flood_lookup,
    on=[
        "district",
        "date"
    ],
    how="left"
)


# ------------------------------------------------------------
# 7. Non-flood days = 0
# ------------------------------------------------------------

training["flood_event"] = (
    training["flood_event"]
    .fillna(0)
    .astype(int)
)


# ------------------------------------------------------------
# 8. Select final columns
# ------------------------------------------------------------

training = training[
    [
        "district",
        "date",
        "rainfall_24h",
        "rainfall_3day",
        "rainfall_7day",
        "flood_event"
    ]
]


# ------------------------------------------------------------
# 9. Sort
# ------------------------------------------------------------

training = training.sort_values(
    [
        "district",
        "date"
    ]
).reset_index(
    drop=True
)


# ------------------------------------------------------------
# 10. Check dataset
# ------------------------------------------------------------

print()
print("Final rows:", len(training))

print()
print("Columns:")

print(
    training.columns.tolist()
)


print()
print("Flood class distribution:")

print(
    training["flood_event"]
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print("Flood percentage:")

print(
    training["flood_event"]
    .mean() * 100
)


print()
print("Rows by district:")

print(
    training["district"]
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print("Missing values:")

print(
    training.isna()
    .sum()
    .to_string()
)


# ------------------------------------------------------------
# 11. Save
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

training.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("Saved:", OUTPUT_FILE)

print()
print("=" * 60)
print("District-day training table created.")
print("=" * 60)