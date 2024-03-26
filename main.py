from Enviroment.simglucose_simulation import Simulation


def main():
    simOne = Simulation().getSimulationResults(24)
    print(simOne)

if __name__ == "__main__":
    main()