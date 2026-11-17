from states import *
from objectiveFunction import *
from display import *
import math
import random

def simulatedAnnealing(mode):
    t = 50
    cooling = 0.999

    currentState = generateRandomState()
    currentCost = calculateCost(currentState, mode)
    costHistory = [currentCost]
    displayBeforeAlgorithm(currentState, currentCost)

    bestState = deepCopyState(currentState)
    bestCost = currentCost

    for i in range(5000):
        neighbor = generateNeighborStateSA(currentState, mode)
        neighborCost = calculateCost(neighbor, mode)
        deltaE = neighborCost - currentCost

        if deltaE < 0:
            currentState = neighbor
            currentCost = neighborCost

        else: #deltaE >= 0
            probability = math.exp(-deltaE/t)

            if random.random() < probability:
                currentState = neighbor
                currentCost = neighborCost   

        if currentCost < bestCost:
            bestState = deepCopyState(currentState)
            bestCost = currentCost
            costHistory.append(bestCost)
            displayDuringAlgorithm(bestCost, i)

        t = t*cooling

        if t < 0.01:
            break

    print()
    return bestState, bestCost, i, costHistory           