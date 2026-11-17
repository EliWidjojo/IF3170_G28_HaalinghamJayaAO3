from readFile import *
from objectiveFunction import *
import random

def generateRandomState():
    state = {
        "warehouse" : [],
        "trucks" : []
    }

    trucks = []
    # create empty trucks
    for vehicle in vehicles:
        trucks.append({
            "vehicle": vehicle["id"],
            "packages" : [],
            "route": []
        })

    # find each truck capacity
    capacity = []
    for vehicle in vehicles:
        capacity.append(vehicle["capacity"])

    # load packages to truck
    random.shuffle(packages)

    for package in packages:
        possibleTrucks = []

        for truckID in range(len(trucks)):
            weight = calculateTruckWeight(trucks[truckID])

            if weight + package["weight"] <= capacity[truckID]:
                possibleTrucks.append(truckID)

        if possibleTrucks:
            truckID = random.choice(possibleTrucks)
            trucks[truckID]["packages"].append(package["id"])
        else:
            state["warehouse"].append(package["id"])

    # Choose route for truck
    for truck in trucks:
        packageIDArr = truck["packages"]
        destinationID = []

        for packageID in packageIDArr:
            destinationID.append(packages[packageID]["destinationID"])

        uniqueDestinations = list(set(destinationID))
        random.shuffle(uniqueDestinations)

        truck["route"] = [0]+ uniqueDestinations + [0]

    state["trucks"] = trucks

    return state

def deepCopyState(state):
    newTrucks = []
    newState = {}
    for truck in state["trucks"]:
        newTrucks.append({
            "vehicle": truck["vehicle"],
            "packages" : truck["packages"].copy(),
            "route" : truck["route"].copy()
        })

    newState["warehouse"] = state["warehouse"].copy()
    newState["trucks"] = newTrucks

    return newState

def chooseSourceTruck(state):
    truckID = random.randrange(len(state["trucks"]))

    while len(state["trucks"][truckID]["packages"]) == 0:
        truckID = random.randrange(len(state["trucks"]))

    return truckID

def chooseSlowestTruck(state):
    i = 0
    slowestTime = -1

    for truck in state["trucks"]:
        t = calculateRouteTime(truck["route"], truck["vehicle"])

        if t > slowestTime and len(truck["packages"]) != 0:
            slowestTime = t
            slowest = i
        i+=1

    return slowest

def rebuildRoute(truck):
    needed = []
    for packageID in truck["packages"]:
        destination = packages[packageID]["destinationID"]
        if destination not in needed:
            needed.append(destination)

    route = truck["route"][1:-1]
    route = [d for d in route if d in needed]

    for destination in needed:
        if destination not in route:
            bestPosition = len(route)
            bestIncrease = float("inf")

            for i in range(len(route)+1):
                temp = route.copy()
                temp.insert(i, destination)

                increase = calculateRouteDistance([0]+temp+[0])

                if increase < bestIncrease:
                    bestIncrease = increase
                    bestPosition = i

            route.insert(bestPosition, destination)

    route = [x for x in route if x != 0]
    truck["route"] = [0] + route + [0]

def movePackageBetweenTrucks(state, mode):
    if mode == "distance":
        truckID1 = chooseSourceTruck(state)
    elif mode == "time":
        truckID1 = chooseSlowestTruck(state)

    packageID = random.choice(state["trucks"][truckID1]["packages"])
    possibleTrucks = []

    for truckID2 in range(len(state["trucks"])):
        if truckID2 == truckID1:
            continue

        truck = state["trucks"][truckID2]

        if calculateTruckWeight(truck) + packages[packageID]["weight"] <= vehicles[truck["vehicle"]]["capacity"]:
            possibleTrucks.append(truckID2)

    if not possibleTrucks:
        return state

    truckID2 = random.choice(possibleTrucks)
    
    state["trucks"][truckID1]["packages"].remove(packageID)
    state["trucks"][truckID2]["packages"].append(packageID)
    rebuildRoute(state["trucks"][truckID1])
    rebuildRoute(state["trucks"][truckID2])

    return state
    
def swapPackageBetweenTrucks(state, mode):
    for i in range(10):
        if mode == "distance":
            truckID1 = chooseSourceTruck(state)
        elif mode == "time":
            truckID1 = chooseSlowestTruck(state)

        truckID2 = random.randrange(len(state["trucks"]))

        while truckID2 == truckID1 or len(state["trucks"][truckID2]["packages"]) == 0:
            truckID2 = random.randrange(len(state["trucks"]))
        
        packageID1 = random.choice(state["trucks"][truckID1]["packages"])
        packageID2 = random.choice(state["trucks"][truckID2]["packages"])

        truck1 = state["trucks"][truckID1]
        truck2 = state["trucks"][truckID2]

        weight1 = calculateTruckWeight(truck1)
        weight2 = calculateTruckWeight(truck2)

        newWeight1 = weight1 - packages[packageID1]["weight"] + packages[packageID2]["weight"]
        newWeight2 = weight2 - packages[packageID2]["weight"] + packages[packageID1]["weight"]
        if newWeight1 <= vehicles[truck1["vehicle"]]["capacity"] and newWeight2 <= vehicles[truck2["vehicle"]]["capacity"]:
            state["trucks"][truckID1]["packages"].remove(packageID1)
            state["trucks"][truckID2]["packages"].remove(packageID2)
            state["trucks"][truckID2]["packages"].append(packageID1)
            state["trucks"][truckID1]["packages"].append(packageID2)

            rebuildRoute(state["trucks"][truckID1])
            rebuildRoute(state["trucks"][truckID2])
            return state
    return state

def movePackageToWarehouse(state, mode):
    if mode == "distance":
        truckID1 = chooseSourceTruck(state)
    elif mode == "time":
        truckID1 = chooseSlowestTruck(state)
    packageID = random.choice(state["trucks"][truckID1]["packages"])
    
    state["trucks"][truckID1]["packages"].remove(packageID)
    state["warehouse"].append(packageID)
    rebuildRoute(state["trucks"][truckID1])
    return state

def movePackageFromWarehouse(state):
    if len(state["warehouse"]) == 0:
        return state
    packageID = random.choice(state["warehouse"])
    possibleTrucks = []

    for truckID2 in range(len(state["trucks"])):
        truck = state["trucks"][truckID2]

        if calculateTruckWeight(truck) + packages[packageID]["weight"] <= vehicles[truck["vehicle"]]["capacity"]:
            possibleTrucks.append(truckID2)

    if not possibleTrucks:
        return state

    truckID2 = random.choice(possibleTrucks)
    
    state["warehouse"].remove(packageID)
    state["trucks"][truckID2]["packages"].append(packageID)
    rebuildRoute(state["trucks"][truckID2])
    return state

def swapRoute(state, mode):
    if mode == "distance":
        truckID = chooseSourceTruck(state)
    elif mode == "time":
        truckID = chooseSlowestTruck(state)

    route = state["trucks"][truckID]["route"]

    if len(route) <= 3:
        return state

    i = random.randint(1, len(route)-2)
    j = random.randint(1, len(route)-2)

    while i == j:
        j = random.randint(1, len(route)-2)   

    route[i], route[j] = route[j], route[i] 

    return state 

def reverseRoute(state, mode):
    if mode == "distance":
        truckID = chooseSourceTruck(state)
    elif mode == "time":
        truckID = chooseSlowestTruck(state)
    route = state["trucks"][truckID]["route"]  
    if len(route) <= 3:
        return state
    i = random.randint(1, len(route)-2)
    j = random.randint(1, len(route)-2)

    while i == j:
        j = random.randint(1, len(route)-2)   

    if i > j:
        i, j = j, i

    route[i:j+1] = reversed(route[i:j+1])

    return state


def generateNeighborStateHC(state, mode):
    chance = random.random()
    neighbor = deepCopyState(state)

    if chance <= 0.35:
        neighbor = reverseRoute(neighbor, mode)

    elif chance <= 0.55:
        neighbor =  swapRoute(neighbor, mode)

    elif chance <= 0.8:
        neighbor = movePackageBetweenTrucks(neighbor, mode)

    elif chance <= 0.9:
        neighbor = movePackageFromWarehouse(neighbor)

    elif chance <= 0.95:
        neighbor = movePackageToWarehouse(neighbor, mode)

    else:
        neighbor = swapPackageBetweenTrucks(neighbor, mode)

    return neighbor

def generateNeighborStateSA(state, mode):
    chance = random.random()
    neighbor = deepCopyState(state)

    if chance <= 0.25:
        neighbor = reverseRoute(neighbor, mode)

    elif chance <= 0.45:
        neighbor =  swapRoute(neighbor, mode)

    elif chance <= 0.75:
        neighbor = movePackageBetweenTrucks(neighbor, mode)

    elif chance <= 0.8:
        neighbor = movePackageFromWarehouse(neighbor)

    elif chance <= 0.9:
        neighbor = movePackageToWarehouse(neighbor, mode)

    else:
        neighbor = swapPackageBetweenTrucks(neighbor, mode)

    return neighbor


def generateNeighborStateGA(state, mode):
    chance = random.random()
    neighbor = deepCopyState(state)

    if chance <= 0.2:
        neighbor = reverseRoute(neighbor, mode)

    elif chance <= 0.4:
        neighbor =  swapRoute(neighbor, mode)

    elif chance <= 0.7:
        neighbor = movePackageBetweenTrucks(neighbor, mode)

    elif chance <= 0.85:
        neighbor = movePackageFromWarehouse(neighbor)

    elif chance <= 0.95:
        neighbor = movePackageToWarehouse(neighbor, mode)

    else:
        neighbor = swapPackageBetweenTrucks(neighbor, mode)

    return neighbor
