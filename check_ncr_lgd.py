import csv
import geopandas as gpd


# --------------------------------------------------
# 1. Load district boundaries
# --------------------------------------------------

boundary_file = "data/raw/boundaries/DISTRICT_BOUNDARY.shp"

gdf = gpd.read_file(boundary_file)


# --------------------------------------------------
# 2. Find Delhi/NCR districts
# --------------------------------------------------

ncr_names = [
    "NEW DELHI",
    "GURUGRAM",
    "FARIDABAD",
    "GHAZIABAD",
    "GAUTAM BUDDHA NAGAR",
    "SONIPAT",
    "PANIPAT"
]

ncr = gdf[
    gdf["DISTRICT"]
    .astype(str)
    .str.upper()
    .isin(ncr_names)
]


print("Delhi/NCR districts found in boundary file:")
print()


for _, row in ncr.iterrows():

    print(
        "District:",
        row["DISTRICT"],
        "| State:",
        row["STATE_UT"],
        "| LGD:",
        row["DIST_LGD"]
    )


# --------------------------------------------------
# 3. Normalize LGD codes
# --------------------------------------------------

def normalize_lgd(code):

    code = str(code).strip()

    if not code:
        return None

    if code.upper() == "NONE":
        return None

    try:
        return str(int(float(code)))
    except ValueError:
        return None


# --------------------------------------------------
# 4. Read historical flood inventory
# --------------------------------------------------

flood_file = "data/raw/India_Flood_Inventory_v3.csv"

flood_codes = set()


with open(
    flood_file,
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        value = row["District_LGD_Codes"].strip()

        if value:

            # One flood event can contain multiple districts
            for code in value.split(","):

                normalized = normalize_lgd(code)

                if normalized is not None:
                    flood_codes.add(normalized)


# --------------------------------------------------
# 5. Compare normalized codes
# --------------------------------------------------

print()
print("Historical flood LGD coverage:")
print()


for _, row in ncr.iterrows():

    boundary_code = normalize_lgd(row["DIST_LGD"])

    if boundary_code in flood_codes:
        status = "MATCHED"
    else:
        status = "NOT FOUND"

    print(
        row["DISTRICT"],
        "| Boundary LGD:", row["DIST_LGD"],
        "| Normalized:", boundary_code,
        "|", status
    )