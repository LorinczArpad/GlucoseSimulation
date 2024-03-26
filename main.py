from Enviroment.simglucose_simulation import Simulation
import pandas as pd


def main():
    simOne = Simulation().getSimulationResults(2)
    edited_df= simOne[simOne['insulin'] != 0]
    edited_df.to_csv('./Results/EditedTestResults.csv', index=True, header=True)
    print('Done')

if __name__ == "__main__":
    main()