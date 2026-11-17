from readFile import *

storeNames = {}

for store in stores:
    storeNames[store["id"]] = store["location"]
    
def calculateDistance(originID, destinationID):
    origin = storeNames[originID]
    destination = storeNames[destinationID]

    return distances[origin][destination]

def calculateDeliveryTime(originID, destinationID):
    origin = storeNames[originID]
    destination = storeNames[destinationID]

    return travelTime[origin][destination]

def calculateServiceTime(place):
    return stores[place]["service_time"]

def calculateRouteDistance(route):
    totalDistance = 0

    for i in range(len(route)-1):
        totalDistance += calculateDistance(route[i], route[i+1])

    return totalDistance

def calculateRouteTime(route, vehicleID):
    totalTime = 0

    for i in range(len(route)-1):
        totalTime += calculateDeliveryTime(route[i], route[i+1])/vehicles[vehicleID]["speed_factor"]
        if route[i+1] != 0:
            totalTime += calculateServiceTime(route[i+1])
    return totalTime

def calculateTotalDistance(solution):
    totalDistance = 0

    for truck in solution["trucks"]:
        totalDistance += calculateRouteDistance(truck["route"])

    return totalDistance


def calculateTotalTime(solution):
    time = []

    for truck in solution["trucks"]:
        time.append(calculateRouteTime(truck["route"], truck["vehicle"]))

    return max(time)

def calculateTruckWeight(truck):
    totalWeight = 0

    for packageID in truck["packages"]:
        totalWeight += packages[packageID]["weight"]

    return totalWeight

def calculateCapacityPenalty(solution):
    penalty = 0

    for truck in solution["trucks"]:
        weight = calculateTruckWeight(truck)

        if weight > vehicles[truck["vehicle"]]["capacity"]:
            excess = max(0, weight - vehicles[truck["vehicle"]]["capacity"])
            penalty += excess * 100

    return penalty

def calculateWarehousePenalty(solution):
    weight = 0

    for warehouse in solution["warehouse"]:
        weight += packages[warehouse]["weight"]

    return weight*20


def calculateCost(solution, mode):
    totalDistance = calculateTotalDistance(solution)
    totalTime = calculateTotalTime(solution)
    if mode == "distance":
        cost = totalDistance
    elif mode == "time":
        cost = totalTime

    cost += calculateCapacityPenalty(solution)
    cost += calculateWarehousePenalty(solution)
    return cost



