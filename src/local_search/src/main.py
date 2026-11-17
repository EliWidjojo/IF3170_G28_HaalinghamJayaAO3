import sys
sys.path.append("../../src/local_search")

from states import *
from hillClimbing import *
from simulatedAnnealing import *
from geneticAlgorithm import *
from createPackages import *
from display import *
import os
def clearScreen():
    os.system("cls" if os.name == "nt" else "clear")

def graph(costHistory):
    print("Enter 'graph' to see Graph!")
    menuGraph = input("> ")
    if menuGraph == "graph":
        makeGraph(costHistory)
        print()
        
loop = True
while loop:
    clearScreen()
    print("What do you want to do?")
    print("1. Hill Climbing")
    print("2. Simulated Annealing")
    print("3. Genetic Algorithm")
    print("4. Randomize packages")
    print("5. Exit")
    menu = int(input("> "))
    clearScreen()
    
    if menu == 1:
        print("Which Hill Climbing?")
        print("1. Steepest - Ascent")
        print("2. Sideways Move")
        print("3. Stochastic")
        print("4. Random Restart")
        menuHill = int(input("> "))
        if menuHill != 1 and menuHill != 2 and menuHill != 3 and menuHill != 4:
            print("ERROR: Please check again!")
            input()
            continue
        clearScreen()
        print("Which mode?")
        print("1. Distance")
        print("2. Time")
        menuMode = int(input("> "))
        if menuMode != 1 and menuMode != 2:
            print("ERROR: Please check again!")
            input()
            continue
        clearScreen()

        modes = {
            1: "distance",
            2: "time"
        }

        hillAlgorithms = {
            1: hillClimbing,
            2: sidewaysMoveHillClimbing,
            3: stochasticHillClimbing,
            4: randomRestartHillClimbing
        }

        mode = modes[menuMode]
        algorithm = hillAlgorithms[menuHill]

        state, cost, i, costHistory = algorithm(mode)
        displayAfterAlgorithm(state, cost, i)

        graph(costHistory)
        input("Enter to clear result!")

    elif menu == 2: 
        print("Which mode?")
        print("1. Distance")
        print("2. Time")
        menuMode = int(input("> "))
        if menuMode != 1 and menuMode != 2:
            print("ERROR: Please check again!")
            input()
            continue
        clearScreen()

        if menuMode == 1:
            state, cost, i, costHistory = simulatedAnnealing("distance")
        else:
            state, cost, i, costHistory = simulatedAnnealing("time")
        displayAfterAlgorithm(state, cost, i)

        graph(costHistory)
        input("Enter to clear result!")
               

    elif menu == 3: 
        print("Which mode?")
        print("1. Distance")
        print("2. Time")
        menuMode = int(input("> "))
        if menuMode != 1 and menuMode != 2:
            print("ERROR: Please check again!")
            input()
            continue
        clearScreen()

        if menuMode == 1:
            state, cost, costHistory = geneticAlgorithm("distance")
        else:
            state, cost, costHistory = geneticAlgorithm("time")

        print("=== FINAL STATE ===")
        displayState(state)
        print("Cost            :", cost)
        print("Total distance  :", calculateTotalDistance(state), "km")
        print("Total time      :", calculateTotalTime(state), "minutes")
        graph(costHistory)
        input("Enter to clear result!")

    elif menu == 4:
        print("How many packages in the warehouse?")
        k = int(input("> "))
        createPackages(k)
        loop = False

    elif menu == 5:
        loop = False
    
    else:
        print("ERROR: Please check again!")
        input()
