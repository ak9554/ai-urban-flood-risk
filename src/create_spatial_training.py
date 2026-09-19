import pandas as pd
from pathlib import Path


RAINFALL_FILE = Path(
    "data/raw/rainfall/ncr_rainfall_2015_2023.csv"
)

FLOOD_FILE = Path(
    "data/processed/ncr_flood_labels_2015_2023.csv"
)

GRID_DISTRICT_FILE = Path(
    "data/processed/ncr_grid_districts.csv"
)

TERRAIN_FILE = Path(
    "data/processed/ncr_terrain_features.csv"
)

LULC_FILE = Path(
    "data/processed/ncr_lulc_features.csv"
)

OUTPUT_FILE = Path(
    "data/processed/ncr_spatial_training.csv"
)


print("=" * 60)
print("Creating efficient spatial training dataset")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load datasets
# ------------------------------------------------------------

rainfall = pd.read_csv(
    RAINFALL_FILE,
    parse_dates=["date"]
)

flood = pd.read_csv(
    FLOOD_FILE,
    parse_dates=["date"]
)

grid_district = pd.read_csv(
    GRID_DISTRICT_FILE
)

terrain = pd.read_csv(
    TERRAIN_FILE
)

lulc = pd.read_csv(
    LULC_FILE
)


# ------------------------------------------------------------
# 2. Normalize district names
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

grid_district["district"] = (
    grid_district["district"]
    .str.strip()
    .str.upper()
)


# ------------------------------------------------------------
# 3. Remove incomplete LULC cells
# ------------------------------------------------------------

lulc_features = [
    "builtup_percent",
    "vegetation_percent",
    "water_percent",
    "cropland_percent",
    "bare_percent"
]

lulc = lulc[
    lulc[lulc_features]
    .notna()
    .all(axis=1)
].copy()


# ------------------------------------------------------------
# 4. Combine static grid features
# ------------------------------------------------------------

grid_features = grid_district.merge(
    terrain,
    on=[
        "grid_id",
        "latitude",
        "longitude"
    ],
    how="inner"
)

grid_features = grid_features.merge(
    lulc[
        [
            "grid_id",
            "builtup_percent",
            "vegetation_percent",
            "water_percent",
            "cropland_percent",
            "bare_percent"
        ]
    ],
    on="grid_id",
    how="inner"
)


print()
print("Usable grid cells:", len(grid_features))


# ------------------------------------------------------------
# 5. Create flood lookup
# ------------------------------------------------------------

flood_lookup = flood[
    [
        "district",
        "date",
        "flood_event"
    ]
].drop_duplicates(
    subset=[
        "district",
        "date"
    ]
)


# ------------------------------------------------------------
# 6. Add flood labels to rainfall
# ------------------------------------------------------------

district_day = rainfall.merge(
    flood_lookup,
    on=[
        "district",
        "date"
    ],
    how="left"
)


district_day["flood_event"] = (
    district_day["flood_event"]
    .fillna(0)
    .astype(int)
)


# ------------------------------------------------------------
# 7. Temporal split
# ------------------------------------------------------------

train_days = district_day[
    district_day["date"] < "2022-01-01"
].copy()

test_days = district_day[
    district_day["date"] >= "2022-01-01"
].copy()


# ------------------------------------------------------------
# 8. Separate positive and negative days
# ------------------------------------------------------------

train_positive = train_days[
    train_days["flood_event"] == 1
].copy()

test_positive = test_days[
    test_days["flood_event"] == 1
].copy()

train_negative = train_days[
    train_days["flood_event"] == 0
].copy()

test_negative = test_days[
    test_days["flood_event"] == 0
].copy()


print()
print("Training flood district-days:", len(train_positive))
print("Testing flood district-days:", len(test_positive))


# ------------------------------------------------------------
# 9. Sample negative days explicitly
# ------------------------------------------------------------

RATIO = 5

train_negative_parts = []

test_negative_parts = []


districts = sorted(
    district_day["district"].unique()
)


for district in districts:

    # -------------------------
    # Training negatives
    # -------------------------

    district_train_positive = train_positive[
        train_positive["district"] == district
    ]

    district_train_negative = train_negative[
        train_negative["district"] == district
    ]

    positive_count = len(
        district_train_positive
    )

    # At least 5 negative days for a district,
    # even if it has no historical positive days.
    requested = max(
        5,
        RATIO * positive_count
    )

    sample_count = min(
        requested,
        len(district_train_negative)
    )

    if sample_count > 0:

        sampled = district_train_negative.sample(
            n=sample_count,
            random_state=42
        )

        train_negative_parts.append(
            sampled
        )


    # -------------------------
    # Testing negatives
    # -------------------------

    district_test_positive = test_positive[
        test_positive["district"] == district
    ]

    district_test_negative = test_negative[
        test_negative["district"] == district
    ]

    positive_count = len(
        district_test_positive
    )

    requested = max(
        5,
        RATIO * positive_count
    )

    sample_count = min(
        requested,
        len(district_test_negative)
    )

    if sample_count > 0:

        sampled = district_test_negative.sample(
            n=sample_count,
            random_state=42
        )

        test_negative_parts.append(
            sampled
        )


# Combine sampled negatives

train_negative_sample = pd.concat(
    train_negative_parts,
    ignore_index=True
)

test_negative_sample = pd.concat(
    test_negative_parts,
    ignore_index=True
)


# ------------------------------------------------------------
# 10. Combine positive and negative days
# ------------------------------------------------------------

train_days_selected = pd.concat(
    [
        train_positive,
        train_negative_sample
    ],
    ignore_index=True
)

test_days_selected = pd.concat(
    [
        test_positive,
        test_negative_sample
    ],
    ignore_index=True
)


# ------------------------------------------------------------
# 11. Check district-day labels BEFORE spatial expansion
# ------------------------------------------------------------

print()
print("=" * 60)
print("SELECTED DISTRICT-DAYS")
print("=" * 60)

print()
print("Training selected:")
print(
    train_days_selected["flood_event"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()
print("Testing selected:")
print(
    test_days_selected["flood_event"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# 12. Attach grid features
# ------------------------------------------------------------

train_spatial = train_days_selected.merge(
    grid_features,
    on="district",
    how="inner"
)

test_spatial = test_days_selected.merge(
    grid_features,
    on="district",
    how="inner"
)


# ------------------------------------------------------------
# 13. Add split labels
# ------------------------------------------------------------

train_spatial["dataset_split"] = "train"

test_spatial["dataset_split"] = "test"


# ------------------------------------------------------------
# 14. Combine train and test
# ------------------------------------------------------------

spatial = pd.concat(
    [
        train_spatial,
        test_spatial
    ],
    ignore_index=True
)


# ------------------------------------------------------------
# 15. Select final columns
# ------------------------------------------------------------

columns = [
    "grid_id",
    "district",
    "date",
    "latitude",
    "longitude",
    "rainfall_24h",
    "rainfall_3day",
    "rainfall_7day",
    "elevation_mean",
    "slope_mean",
    "builtup_percent",
    "vegetation_percent",
    "water_percent",
    "cropland_percent",
    "bare_percent",
    "flood_event",
    "dataset_split"
]


spatial = spatial[
    columns
]


# ------------------------------------------------------------
# 16. Sort
# ------------------------------------------------------------

spatial = (
    spatial
    .sort_values(
        [
            "dataset_split",
            "date",
            "grid_id"
        ]
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 17. Final checks
# ------------------------------------------------------------

print()
print("=" * 60)
print("SPATIAL TRAINING DATASET")
print("=" * 60)

print()
print("Rows:", len(spatial))

print(
    "Unique grid cells:",
    spatial["grid_id"].nunique()
)

print(
    "Unique dates:",
    spatial["date"].nunique()
)


print()
print("Rows by split:")

print(
    spatial["dataset_split"]
    .value_counts()
    .to_string()
)


print()
print("Flood labels:")

print(
    spatial["flood_event"]
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print("Flood labels by split:")

print(
    pd.crosstab(
        spatial["dataset_split"],
        spatial["flood_event"]
    )
)


print()
print("Missing values:")

print(
    spatial.isna()
    .sum()
    .to_string()
)


print()
print("Unique grid cells by split:")

print(
    spatial.groupby(
        "dataset_split"
    )["grid_id"]
    .nunique()
    .to_string()
)


# ------------------------------------------------------------
# 18. Save
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

spatial.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("Saved:", OUTPUT_FILE)

print()
print("=" * 60)
print("Spatial training dataset created.")
print("=" * 60)