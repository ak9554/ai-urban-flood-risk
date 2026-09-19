import geopandas as gpd
from pathlib import Path


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BOUNDARY_FILE = Path(
    "data/raw/boundaries/DISTRICT_BOUNDARY.shp"
)

OUTPUT_FILE = Path(
    "data/processed/ncr_7_districts.geojson"
)


# --------------------------------------------------
# 2. Our 7-district NCR study area
# --------------------------------------------------

NCR_DISTRICTS = [
    "NEW DELHI",
    "GURUGRAM",
    "FARIDABAD",
    "GHAZIABAD",
    "GAUTAM BUDDHA NAGAR",
    "SONIPAT",
    "PANIPAT",
]


# --------------------------------------------------
# 3. Load Survey of India boundaries
# --------------------------------------------------

print("Loading district boundaries...")

gdf = gpd.read_file(BOUNDARY_FILE)

print(f"Total districts in boundary file: {len(gdf)}")


# --------------------------------------------------
# 4. Clean district names
# --------------------------------------------------

gdf["DISTRICT_CLEAN"] = (
    gdf["DISTRICT"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# --------------------------------------------------
# 5. Select our 7 NCR districts
# --------------------------------------------------

ncr = gdf[
    gdf["DISTRICT_CLEAN"].isin(NCR_DISTRICTS)
].copy()


# --------------------------------------------------
# 6. Check that all 7 were found
# --------------------------------------------------

found = sorted(ncr["DISTRICT_CLEAN"].unique())

print()
print("NCR districts found:")
for district in found:
    print(f"  - {district}")


missing = sorted(
    set(NCR_DISTRICTS) - set(found)
)


if missing:
    print()
    print("ERROR: These districts were not found:")

    for district in missing:
        print(f"  - {district}")

    raise SystemExit(
        "Stopping because the NCR boundary is incomplete."
    )


# --------------------------------------------------
# 7. Convert to WGS84
# --------------------------------------------------
#
# WGS84 = latitude/longitude coordinates
# EPSG:4326 is the standard geographic CRS.
#

ncr = ncr.to_crs(epsg=4326)


# --------------------------------------------------
# 8. Keep useful columns
# --------------------------------------------------

ncr = ncr[
    [
        "DISTRICT",
        "DIST_LGD",
        "STATE_UT",
        "geometry",
    ]
]


# --------------------------------------------------
# 9. Save GeoJSON
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

ncr.to_file(
    OUTPUT_FILE,
    driver="GeoJSON"
)


# --------------------------------------------------
# 10. Verify the saved file
# --------------------------------------------------

print()
print("SUCCESS!")
print(f"Saved to: {OUTPUT_FILE}")

print()
print(f"Number of NCR districts: {len(ncr)}")

print()
print("Final districts:")

for district in ncr["DISTRICT"]:
    print(f"  - {district}")


print()
print(
    f"Output file size: "
    f"{OUTPUT_FILE.stat().st_size:,} bytes"
)