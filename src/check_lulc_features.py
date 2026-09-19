import pandas as pd
from pathlib import Path


LULC_FILE = Path(
    "data/processed/ncr_lulc_features.csv"
)


print("=" * 60)
print("LULC feature quality check")
print("=" * 60)


df = pd.read_csv(LULC_FILE)


# ------------------------------------------------------------
# Basic information
# ------------------------------------------------------------

print("Total rows:", len(df))

print()
print("Columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

feature_columns = [
    "builtup_percent",
    "vegetation_percent",
    "water_percent",
    "cropland_percent",
    "bare_percent"
]

print()
print("Missing values:")
print(df[feature_columns].isna().sum())


# ------------------------------------------------------------
# Check percentage ranges
# ------------------------------------------------------------

print()
print("Checking percentage ranges...")

for column in feature_columns:

    minimum = df[column].min()
    maximum = df[column].max()

    print(
        f"{column}: "
        f"min={minimum:.2f}, "
        f"max={maximum:.2f}"
    )

    if minimum < 0 or maximum > 100:
        print("WARNING: value outside 0-100 range")


# ------------------------------------------------------------
# Check duplicate grid IDs
# ------------------------------------------------------------

duplicates = df["grid_id"].duplicated().sum()

print()
print("Duplicate grid IDs:", duplicates)


# ------------------------------------------------------------
# Count usable cells
# ------------------------------------------------------------

usable = df[feature_columns].notna().all(axis=1)

print()
print("Usable LULC cells:", usable.sum())
print("Excluded LULC cells:", (~usable).sum())


# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

if (
    usable.sum() == 8471
    and
    (~usable).sum() == 3
    and
    duplicates == 0
):
    print()
    print("RESULT: LULC dataset passed quality checks.")
else:
    print()
    print("RESULT: Review the LULC dataset.")


print()
print("=" * 60)
print("LULC quality check complete.")
print("=" * 60)