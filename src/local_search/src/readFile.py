import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

with open(os.path.join(DATA_DIR, "distance.json"), 'r') as file:
    distances = json.load(file)

with open(os.path.join(DATA_DIR, "time.json"), 'r') as file:
    travelTime = json.load(file)

with open(os.path.join(DATA_DIR, "stores.json"), 'r') as file:
    stores = json.load(file)

with open(os.path.join(DATA_DIR, "packages.json"), 'r') as file:
    packages = json.load(file)

with open(os.path.join(DATA_DIR, "vehicles.json"), 'r') as file:
    vehicles = json.load(file)




