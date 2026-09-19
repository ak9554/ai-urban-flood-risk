import pandas as pd
from pathlib import Path
import re


FLOOD_FILE = Path(
    "data/raw/India_Flood_Inventory_v3.csv"
)


print("=" * 60)
print("Checking unusually long NCR flood events")
print("=" * 60)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(FLOOD_FILE)


# ------------------------------------------------------------
# NCR LGD codes
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
# Parse dates
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
# Select NCR events in 2015-2023
# ------------------------------------------------------------

ncr = df[
    df["NCR_districts"].apply(len) > 0
].copy()

ncr = ncr[
    (ncr["start_date"] >= "2015-01-01")
    &
    (ncr["start_date"] <= "2023-12-31")
].copy()


# ------------------------------------------------------------
# Calculate actual duration from dates
# ------------------------------------------------------------

ncr["calculated_duration"] = (
    ncr["end_date"] - ncr["start_date"]
).dt.days + 1


# ------------------------------------------------------------
# Show all events longer than 14 days
# ------------------------------------------------------------

long_events = ncr[
    ncr["calculated_duration"] > 14
].copy()


print()
print(
    "Events longer than 14 days:",
    len(long_events)
)


columns = [
    "UEI",
    "start_date",
    "end_date",
    "calculated_duration",
    "NCR_districts",
    "Main Cause"
]


print()

if len(long_events) > 0:

    print(
        long_events[
            columns
        ]
        .sort_values("calculated_duration", ascending=False)
        .to_string(index=False)
    )

else:

    print("None")


# ------------------------------------------------------------
# Duration statistics
# ------------------------------------------------------------

print()
print("Duration statistics:")

print(
    ncr["calculated_duration"]
    .describe()
)


print()
print("=" * 60)
print("Long-event inspection complete.")
print("=" * 60)