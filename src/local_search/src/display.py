from readFile import *
from objectiveFunction import *

import matplotlib.pyplot as plt

def makeGraph(costHistory):
    plt.plot(costHistory)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Objective Function Progress")
    plt.show()

def displayState(state):
    print("Undelivered Packages: ", state["warehouse"])
    print("=============================")
    for truck in state["trucks"]:
        print("Vehicle ID      :", truck["vehicle"])
        print("Vehicle Type    :", vehicles[truck["vehicle"]]["type"])
        print("Package ID      :", truck["packages"])
        print("Route           :", end = " ")
        i = 0
        for route in truck["route"]:
            print(stores[route]["location"], end="")

            if i != len(truck["route"]) - 1:
                print(" - ", end="")
            i+=1
        print()
        print()

def displayBeforeAlgorithm(state, cost):
    print("=== INITIAL STATE ===")
    displayState(state)
    print("Cost            :", cost)
    print()


def displayDuringAlgorithm(cost, iteration):
    print("Iteration:", iteration+1, "| Cost:", cost)
 
def displayAfterAlgorithm(state, cost, iteration):
    print("=== FINAL STATE ===")
    displayState(state)
    print("Cost            :", cost)
    print("Total iteration :", iteration+1)
    print("Total distance  :", calculateTotalDistance(state), "km")
    print("Total time      :", calculateTotalTime(state), "minutes")
    print()


