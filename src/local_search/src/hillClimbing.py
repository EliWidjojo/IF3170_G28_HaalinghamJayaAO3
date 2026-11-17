from states import *
from objectiveFunction import *
from display import *

def hillClimbing(mode):
    currentState = generateRandomState()
    currentCost = calculateCost(currentState, mode)
    costHistory = [currentCost]
    displayBeforeAlgorithm(currentState, currentCost)

    for i in range(100):
        bestNeighbor = None
        bestCost = float("inf")
        for j in range(50):
            neighbor = generateNeighborStateHC(currentState, mode)
            cost = calculateCost(neighbor, mode)

            if cost < bestCost:
                bestNeighbor = neighbor
                bestCost = cost

        if bestCost < currentCost:
            currentState = bestNeighbor
            currentCost = bestCost
            displayDuringAlgorithm(currentCost, i)
            costHistory.append(currentCost)
        else:
            break

    print()
    return currentState, currentCost, i, costHistory

def hillClimbingNoPrint(mode):
    currentState = generateRandomState()
    currentCost = calculateCost(currentState, mode)

    for i in range(100):
        bestNeighbor = None
        bestCost = float("inf")
        for j in range(50):
            neighbor = generateNeighborStateHC(currentState, mode)
            cost = calculateCost(neighbor, mode)

            if cost < bestCost:
                bestNeighbor = neighbor
                bestCost = cost

        if bestCost < currentCost:
            currentState = bestNeighbor
            currentCost = bestCost
        else:
            break

    print()
    return currentState, currentCost, i

def sidewaysMoveHillClimbing(mode):
    currentState = generateRandomState()
    currentCost = calculateCost(currentState, mode)
    costHistory = [currentCost]
    noImprovement = 0
    displayBeforeAlgorithm(currentState, currentCost)

    for i in range(100):
        bestNeighbor = None
        bestCost = float("inf")
        for j in range(50):
            neighbor = generateNeighborStateHC(currentState, mode)
            cost = calculateCost(neighbor, mode)

            if cost < bestCost:
                bestNeighbor = neighbor
                bestCost = cost

        if bestCost < currentCost:
            currentState = bestNeighbor
            currentCost = bestCost
            noImprovement = 0
            costHistory.append(currentCost)
            displayDuringAlgorithm(currentCost, i)
        elif bestCost == currentCost:
            currentState = bestNeighbor
            currentCost = bestCost
            noImprovement += 1
            costHistory.append(currentCost)  
            displayDuringAlgorithm(currentCost, i)         
        else:
            noImprovement+=1

        if noImprovement >= 5:
            break

    print()
    return currentState, currentCost, i, costHistory

def stochasticHillClimbing(mode):
    currentState = generateRandomState()
    currentCost = calculateCost(currentState, mode)
    costHistory = [currentCost]
    displayBeforeAlgorithm(currentState, currentCost)

    for i in range(100):
        neighbors = []
        for j in range(50):
            neighbor = generateNeighborStateHC(currentState, mode)
            cost = calculateCost(neighbor, mode)

            neighbors.append((neighbor, cost))

        improving = [n for n in neighbors if n[1] < currentCost]
        if not improving:
            break
        currentState, currentCost = random.choice(improving)
        displayDuringAlgorithm(currentCost, i)
        costHistory.append(currentCost)

    print()
    return currentState, currentCost, i, costHistory    
        
def randomRestartHillClimbing(mode):
    bestCost = float('inf')
    bestState = None
    besti = 0
    costHistory = []
    
    for i in range(100):
        currentState, currentCost, currenti= hillClimbingNoPrint(mode)
        print("Restart", i + 1)
        print("Cost            :", currentCost)
        print("HC Iteration    :", currenti + 1)
        print()
        if currentCost < bestCost:
            bestCost = currentCost
            bestState = currentState
            besti = currenti
        costHistory.append(currentCost)

    return bestState, bestCost, besti, costHistory
