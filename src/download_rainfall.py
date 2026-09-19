import requests
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

OUTPUT_DIR = Path("data/raw/rainfall")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "ncr_rainfall_2015_2023.csv"


# --------------------------------------------------
# 2. Our 7-district NCR study area
# --------------------------------------------------

DISTRICTS = {
    "New Delhi": (28.61, 77.23),
    "Gurugram": (28.46, 77.03),
    "Faridabad": (28.41, 77.31),
    "Ghaziabad": (28.67, 77.42),
    "Gautam Buddha Nagar": (28.57, 77.32),
    "Sonipat": (28.99, 77.02),
    "Panipat": (29.39, 76.97),
}


# --------------------------------------------------
# 3. Open-Meteo historical API
# --------------------------------------------------

URL = "https://archive-api.open-meteo.com/v1/archive"

start_date = "2015-01-01"
end_date = "2023-12-31"

latitudes = list(DISTRICTS.values())
latitudes = [x[0] for x in latitudes]

longitudes = list(DISTRICTS.values())
longitudes = [x[1] for x in longitudes]


params = {
    "latitude": latitudes,
    "longitude": longitudes,
    "start_date": start_date,
    "end_date": end_date,
    "daily": "precipitation_sum",
    "timezone": "Asia/Kolkata",
}


# --------------------------------------------------
# 4. Request rainfall data
# --------------------------------------------------

print("Downloading rainfall data...")
print(f"Period: {start_date} to {end_date}")
print(f"Districts: {len(DISTRICTS)}")

response = requests.get(
    URL,
    params=params,
    timeout=120
)

response.raise_for_status()

data = response.json()


# --------------------------------------------------
# 5. Convert API response into a dataframe
# --------------------------------------------------

district_names = list(DISTRICTS.keys())

all_data = []

for i, district in enumerate(district_names):

    district_data = data[i]["daily"]

    df = pd.DataFrame({
        "date": district_data["time"],
        "rainfall_mm": district_data["precipitation_sum"],
    })

    df["district"] = district

    all_data.append(df)


rainfall = pd.concat(
    all_data,
    ignore_index=True
)


# --------------------------------------------------
# 6. Make sure dates are properly formatted
# --------------------------------------------------

rainfall["date"] = pd.to_datetime(rainfall["date"])

rainfall = rainfall.sort_values(
    ["district", "date"]
).reset_index(drop=True)


# --------------------------------------------------
# 7. Create rainfall features
# --------------------------------------------------

# Rainfall during the current day
rainfall["rainfall_24h"] = rainfall["rainfall_mm"]


# Total rainfall during current + previous 2 days
rainfall["rainfall_3day"] = (
    rainfall
    .groupby("district")["rainfall_mm"]
    .rolling(3, min_periods=1)
    .sum()
    .reset_index(level=0, drop=True)
)


# Total rainfall during current + previous 6 days
rainfall["rainfall_7day"] = (
    rainfall
    .groupby("district")["rainfall_mm"]
    .rolling(7, min_periods=1)
    .sum()
    .reset_index(level=0, drop=True)
)


# --------------------------------------------------
# 8. Keep useful columns
# --------------------------------------------------

rainfall = rainfall[
    [
        "district",
        "date",
        "rainfall_24h",
        "rainfall_3day",
        "rainfall_7day",
    ]
]


# --------------------------------------------------
# 9. Save the dataset
# --------------------------------------------------

rainfall.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 10. Basic verification
# --------------------------------------------------

print()
print("Rainfall download successful!")
print(f"Rows: {len(rainfall):,}")
print(f"Saved to: {OUTPUT_FILE}")

print()
print("Districts:")
print(rainfall["district"].unique())

print()
print("Date range:")
print(rainfall["date"].min())
print(rainfall["date"].max())

print()
print("First 10 rows:")
print(rainfall.head(10))