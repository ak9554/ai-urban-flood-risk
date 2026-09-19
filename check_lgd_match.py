import csv

flood_file = "data/raw/India_Flood_Inventory_v3.csv"

codes = set()

with open(flood_file, encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        value = row["District_LGD_Codes"].strip()

        if value:
            codes.add(value)

print("Unique District_LGD_Codes in flood inventory:", len(codes))

print("\nFirst 30 codes:")
for code in sorted(codes)[:30]:
    print(code)