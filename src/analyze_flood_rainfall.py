import pandas as pd
from pathlib import Path


TRAINING_FILE = Path(
    "data/processed/ncr_district_day_training.csv"
)


print("=" * 60)
print("Analyzing rainfall on historical flood vs non-flood days")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

df = pd.read_csv(
    TRAINING_FILE
)


# ------------------------------------------------------------
# 2. Split classes
# ------------------------------------------------------------

flood_days = df[
    df["flood_event"] == 1
]

non_flood_days = df[
    df["flood_event"] == 0
]


print()
print("Flood days:", len(flood_days))
print("Non-flood days:", len(non_flood_days))


# ------------------------------------------------------------
# 3. Compare rainfall statistics
# ------------------------------------------------------------

rainfall_columns = [
    "rainfall_24h",
    "rainfall_3day",
    "rainfall_7day"
]


for column in rainfall_columns:

    print()
    print("-" * 60)
    print(column)
    print("-" * 60)

    print()
    print("Flood days:")

    print(
        flood_days[column]
        .describe()
        .to_string()
    )

    print()
    print("Non-flood days:")

    print(
        non_flood_days[column]
        .describe()
        .to_string()
    )


# ------------------------------------------------------------
# 4. Mean comparison
# ------------------------------------------------------------

print()
print("=" * 60)
print("Mean rainfall comparison")
print("=" * 60)

for column in rainfall_columns:

    flood_mean = flood_days[column].mean()

    non_flood_mean = non_flood_days[column].mean()

    print(
        f"{column}: "
        f"Flood={flood_mean:.2f} mm | "
        f"Non-flood={non_flood_mean:.2f} mm"
    )


# ------------------------------------------------------------
# 5. High rainfall thresholds
# ------------------------------------------------------------

thresholds = [
    10,
    20,
    30,
    50,
    75,
    100
]


print()
print("=" * 60)
print("Flood-day rainfall threshold analysis")
print("=" * 60)


for threshold in thresholds:

    count_24h = (
        flood_days["rainfall_24h"] >= threshold
    ).sum()

    total_flood = len(flood_days)

    percentage = (
        count_24h / total_flood
    ) * 100

    print(
        f"24h rainfall >= {threshold} mm: "
        f"{count_24h}/{total_flood} "
        f"flood days ({percentage:.1f}%)"
    )


# ------------------------------------------------------------
# 6. Highest rainfall flood days
# ------------------------------------------------------------

print()
print("=" * 60)
print("Highest rainfall historical flood days")
print("=" * 60)

print(
    flood_days[
        [
            "district",
            "date",
            "rainfall_24h",
            "rainfall_3day",
            "rainfall_7day",
            "flood_event"
        ]
    ]
    .sort_values(
        "rainfall_24h",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)


print()
print("=" * 60)
print("Rainfall analysis complete.")
print("=" * 60)