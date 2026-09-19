import csv
import geopandas as gpd

# Load district boundaries
boundary_file = "data/raw/boundaries/DISTRICT_BOUNDARY.shp"
gdf = gpd.read_file(boundary_file)

# Get valid boundary LGD codes
boundary_codes = set(
    str(code).strip()
    for code in gdf["DIST_LGD"]
    if str(code).strip() != "NOT AVAILABLE"
)

# Read flood inventory
flood_file = "data/raw/India_Flood_Inventory_v3.csv"

flood_codes = set()

with open(flood_file, encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        value = row["District_LGD_Codes"].strip()

        if value:
            # One flood event can contain multiple district codes
            for code in value.split(","):
                code = code.strip()

                if code:
                    flood_codes.add(code)

# Compare the two sets
matched = flood_codes & boundary_codes
unmatched = flood_codes - boundary_codes

print("Flood inventory unique individual LGD codes:", len(flood_codes))
print("Boundary LGD codes:", len(boundary_codes))
print("Matched LGD codes:", len(matched))
print("Unmatched LGD codes:", len(unmatched))

print("\nFirst 30 matched codes:")

for code in sorted(matched, key=lambda x: int(x)):
    print(code)

    if code == sorted(matched, key=lambda x: int(x))[29]:
        break

print("\nFirst 30 unmatched codes:")

count = 0

for code in unmatched:
    print(code)
    count += 1

    if count == 30:
        break