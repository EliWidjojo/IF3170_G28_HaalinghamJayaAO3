from states import *
from objectiveFunction import *
import random

def tournamentSelection(population, mode):
    candidates = random.sample(population, 5)
    best = candidates[0]

    for candidate in candidates:
        if calculateCost(candidate, mode) < calculateCost(best, mode):
            best = candidate

    return best

def crossover(parent1, parent2):
    child = deepCopyState(parent1)

    for i in range(len(child["trucks"])):
        if random.random() < 0.5:
            child["trucks"][i]["packages"] = parent2["trucks"][i]["packages"].copy()

        rebuildRoute(child["trucks"][i])

    return child

def mutate(state, mode):
    if random.random() < 0.1:
        state = generateNeighborStateGA(state, mode)
    return state

def geneticAlgorithm(mode):
    population = []
    costHistory = []

    for i in range(100):
        population.append(generateRandomState())

    print("=== INITIAL STATE ===")
    best = min(population, key=lambda x: calculateCost(x, mode))
    bestCost = calculateCost(best, mode)
    print("Best Cost       :", bestCost)
    print()

    for generation in range(100):
        newPopulation = []

        while len(newPopulation) < len(population):
            parent1 = tournamentSelection(population, mode)
            parent2 = tournamentSelection(population, mode)

            child = crossover(parent1, parent2)
            child = mutate(child, mode)

            newPopulation.append(child)

        population = newPopulation
        best = min(population, key=lambda x: calculateCost(x, mode))
        bestCost = calculateCost(best, mode)

        costHistory.append(bestCost)

        print("Generation:", generation+1, "| Cost:", bestCost)
        print()

    best = min(population, key = lambda x : calculateCost(x, mode))
    bestCost = calculateCost(best, mode)
    

    return best, bestCost, costHistory