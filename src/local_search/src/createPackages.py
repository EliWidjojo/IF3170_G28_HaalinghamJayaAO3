import json
import random
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def createPackages(k):
    with open(os.path.join(DATA_DIR, "stores.json"), 'r') as file:
        stores = json.load(file)

    packages = []
    storeNames = {}

    for store in stores:
        storeNames[store["id"]] = store["location"]


    for i in range(k):
        packages.append({
            "id": i,
            "destinationID": random.choice(list(storeNames.keys())[1:]),
            "weight": random.randint(5, 100),
        })

    with open(os.path.join(DATA_DIR, "packages.json"), 'w') as f:
        json.dump(packages, f, indent=4)

    print("Created Packages!")
    print("Please restart the program to load the new data.")
    
