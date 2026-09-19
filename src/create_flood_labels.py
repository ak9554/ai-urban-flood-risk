import pandas as pd
import re
from pathlib import Path


FLOOD_FILE = Path(
    "data/raw/India_Flood_Inventory_v3.csv"
)

OUTPUT_FILE = Path(
    "data/processed/ncr_flood_labels_2015_2023.csv"
)


print("=" * 60)
print("Creating cleaned NCR historical flood labels")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load flood inventory
# ------------------------------------------------------------

df = pd.read_csv(FLOOD_FILE)


# ------------------------------------------------------------
# 2. NCR district LGD codes
# ------------------------------------------------------------

ncr_lgd = {
    94: "NEW DELHI",
    86: "GURUGRAM",
    88: "FARIDABAD",
    75: "PANIPAT",
    76: "SONIPAT",
    141: "GAUTAM BUDDHA NAGAR",
    140: "GHAZIABAD"
}


# ------------------------------------------------------------
# 3. Extract LGD codes
# ------------------------------------------------------------

def extract_lgd_codes(value):

    if pd.isna(value):
        return []

    return [
        int(x)
        for x in re.findall(r"\d+", str(value))
    ]


df["LGD_code_list"] = df[
    "District_LGD_Codes"
].apply(extract_lgd_codes)


# ------------------------------------------------------------
# 4. Find NCR districts
# ------------------------------------------------------------

def find_ncr_districts(codes):

    return [
        ncr_lgd[code]
        for code in codes
        if code in ncr_lgd
    ]


df["NCR_districts"] = df[
    "LGD_code_list"
].apply(find_ncr_districts)


# ------------------------------------------------------------
# 5. Parse dates
# ------------------------------------------------------------

df["start_date"] = pd.to_datetime(
    df["Start Date"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

df["end_date"] = pd.to_datetime(
    df["End Date"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)


# ------------------------------------------------------------
# 6. Keep NCR events in rainfall period
# ------------------------------------------------------------

ncr = df[
    df["NCR_districts"].apply(len) > 0
].copy()

ncr = ncr[
    (ncr["start_date"] >= "2015-01-01")
    &
    (ncr["start_date"] <= "2023-12-31")
].copy()


print("NCR events in 2015-2023:", len(ncr))


# ------------------------------------------------------------
# 7. Calculate duration
# ------------------------------------------------------------

ncr["calculated_duration"] = (
    ncr["end_date"] - ncr["start_date"]
).dt.days + 1


# ------------------------------------------------------------
# 8. Remove unusually long records
# ------------------------------------------------------------

MAX_EVENT_DURATION = 14

long_events = ncr[
    ncr["calculated_duration"] > MAX_EVENT_DURATION
].copy()

ncr_clean = ncr[
    ncr["calculated_duration"] <= MAX_EVENT_DURATION
].copy()


print()
print(
    "Excluded events (>14 days):",
    len(long_events)
)

print(
    "Events retained:",
    len(ncr_clean)
)


print()
print("Excluded events:")

if len(long_events) > 0:

    print(
        long_events[
            [
                "UEI",
                "start_date",
                "end_date",
                "calculated_duration",
                "NCR_districts"
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# 9. Create district-date labels
# ------------------------------------------------------------

records = []

for _, event in ncr_clean.iterrows():

    dates = pd.date_range(
        start=event["start_date"],
        end=event["end_date"],
        freq="D"
    )

    for district in event["NCR_districts"]:

        for date in dates:

            records.append({
                "district": district,
                "date": date.date(),
                "flood_event": 1,
                "event_id": event["UEI"]
            })


labels = pd.DataFrame(records)


# ------------------------------------------------------------
# 10. Remove duplicate district-date combinations
# ------------------------------------------------------------

labels = (
    labels
    .sort_values(["district", "date"])
    .drop_duplicates(
        subset=["district", "date"]
    )
)


# ------------------------------------------------------------
# 11. Summary
# ------------------------------------------------------------

print()
print(
    "Unique district-date flood labels:",
    len(labels)
)

print()
print("Labels by district:")

print(
    labels["district"]
    .value_counts()
    .to_string()
)


print()
print("Date range:")

print("First:", labels["date"].min())
print("Last:", labels["date"].max())


# ------------------------------------------------------------
# 12. Save
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

labels.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("Saved:", OUTPUT_FILE)

print()
print("=" * 60)
print("Clean flood label creation complete.")
print("=" * 60)