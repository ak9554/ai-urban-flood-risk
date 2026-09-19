import csv
import os

files = [
    "District_FloodedArea.csv",
    "District_FloodImpact.csv"
]

terms = [
    "Delhi",
    "Gurgaon",
    "Gurugram",
    "Faridabad",
    "Ghaziabad",
    "Noida",
    "Gautam",
    "Sonepat",
    "Sonipat",
    "Panipat",
    "Bahadurgarh"
]

for filename in files:
    path = os.path.join("data", "raw", filename)

    print("\n###", filename)

    with open(path, encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            text = " ".join(row).lower()

            if any(term.lower() in text for term in terms):
                print(row)