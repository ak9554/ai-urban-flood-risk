import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Load rainfall dataset
# --------------------------------------------------

FILE = Path("data/raw/rainfall/ncr_rainfall_2015_2023.csv")

rainfall = pd.read_csv(FILE)

rainfall["date"] = pd.to_datetime(rainfall["date"])


# --------------------------------------------------
# 2. Basic information
# --------------------------------------------------

print("=" * 60)
print("RAINFALL DATASET CHECK")
print("=" * 60)

print(f"\nTotal rows: {len(rainfall):,}")

print("\nColumns:")
print(list(rainfall.columns))


# --------------------------------------------------
# 3. Check districts
# --------------------------------------------------

print("\nDistricts:")
print(rainfall["district"].value_counts())


# --------------------------------------------------
# 4. Check date coverage
# --------------------------------------------------

print("\nDate range:")
print("Start:", rainfall["date"].min())
print("End:  ", rainfall["date"].max())


# --------------------------------------------------
# 5. Check missing values
# --------------------------------------------------

print("\nMissing values:")
print(rainfall.isna().sum())


# --------------------------------------------------
# 6. Check negative rainfall
# --------------------------------------------------

negative_values = (
    rainfall[
        (rainfall["rainfall_24h"] < 0)
        | (rainfall["rainfall_3day"] < 0)
        | (rainfall["rainfall_7day"] < 0)
    ]
)

print("\nNegative rainfall rows:", len(negative_values))


# --------------------------------------------------
# 7. Rainfall statistics
# --------------------------------------------------

print("\nRainfall statistics:")
print(
    rainfall[
        [
            "rainfall_24h",
            "rainfall_3day",
            "rainfall_7day",
        ]
    ].describe()
)


# --------------------------------------------------
# 8. Statistics by district
# --------------------------------------------------

print("\nMaximum 24-hour rainfall by district:")

max_rain = (
    rainfall
    .groupby("district")["rainfall_24h"]
    .max()
    .sort_values(ascending=False)
)

print(max_rain)


# --------------------------------------------------
# 9. Check duplicate district/date combinations
# --------------------------------------------------

duplicates = rainfall.duplicated(
    subset=["district", "date"]
).sum()

print("\nDuplicate district/date rows:", duplicates)


# --------------------------------------------------
# 10. Final result
# --------------------------------------------------

print("\n" + "=" * 60)

if (
    rainfall.isna().sum().sum() == 0
    and len(negative_values) == 0
    and duplicates == 0
):
    print("RESULT: Rainfall dataset passed basic quality checks.")
else:
    print("RESULT: Some data-quality issues need investigation.")

print("=" * 60)