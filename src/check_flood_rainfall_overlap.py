import pandas as pd
from pathlib import Path


FLOOD_FILE = Path(
    "data/processed/ncr_flood_labels_2015_2023.csv"
)

RAINFALL_FILE = Path(
    "data/raw/rainfall/ncr_rainfall_2015_2023.csv"
)


print("=" * 60)
print("Checking flood-label and rainfall overlap")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load datasets
# ------------------------------------------------------------

flood = pd.read_csv(FLOOD_FILE)

rainfall = pd.read_csv(RAINFALL_FILE)


# ------------------------------------------------------------
# 2. Convert dates
# ------------------------------------------------------------

flood["date"] = pd.to_datetime(
    flood["date"]
)

rainfall["date"] = pd.to_datetime(
    rainfall["date"]
)


# ------------------------------------------------------------
# 3. Normalize district names
# ------------------------------------------------------------

flood["district"] = (
    flood["district"]
    .str.strip()
    .str.upper()
)

rainfall["district"] = (
    rainfall["district"]
    .str.strip()
    .str.upper()
)


# ------------------------------------------------------------
# 4. Check date ranges
# ------------------------------------------------------------

print()
print("Rainfall date range:")
print("First:", rainfall["date"].min())
print("Last:", rainfall["date"].max())


print()
print("Flood label date range:")
print("First:", flood["date"].min())
print("Last:", flood["date"].max())


# ------------------------------------------------------------
# 5. Merge
# ------------------------------------------------------------

merged = flood.merge(
    rainfall,
    on=["district", "date"],
    how="left"
)


# ------------------------------------------------------------
# 6. Check matches
# ------------------------------------------------------------

matched = merged[
    merged["rainfall_24h"].notna()
]

unmatched = merged[
    merged["rainfall_24h"].isna()
]


print()
print("Total flood labels:", len(flood))

print(
    "Flood labels with rainfall:",
    len(matched)
)

print(
    "Flood labels without rainfall:",
    len(unmatched)
)


# ------------------------------------------------------------
# 7. Show unmatched records
# ------------------------------------------------------------

if len(unmatched) > 0:

    print()
    print("Unmatched flood labels:")

    print(
        unmatched[
            ["district", "date"]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# 8. Show matched rainfall values
# ------------------------------------------------------------

print()
print("Rainfall values for matched flood dates:")

if len(matched) > 0:

    print(
        matched[
            [
                "district",
                "date",
                "rainfall_24h",
                "rainfall_3day",
                "rainfall_7day"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

else:

    print("No matches found.")


# ------------------------------------------------------------
# 9. District overlap
# ------------------------------------------------------------

print()
print("Flood labels by district:")

print(
    flood["district"]
    .value_counts()
    .to_string()
)


print()
print("Rainfall records by district:")

print(
    rainfall["district"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 10. Final check
# ------------------------------------------------------------

print()
print("=" * 60)

if len(unmatched) == 0:

    print(
        "SUCCESS: Every flood label has rainfall data."
    )

else:

    print(
        "WARNING:",
        len(unmatched),
        "flood labels are still unmatched."
    )

print("=" * 60)