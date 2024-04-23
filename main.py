from Enviroment.simglucose_simulation import Simulation
import pandas as pd
import random
def UseSimulation():
    data = pd.read_csv('C:\\Users\\dudu\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\simglucose\\params\\vpatient_params.csv')
    names = data['Name']

    for name in names:
        if(name != 'patient1'): 
         simOne = Simulation(name).getSimulationResults(1)
         edited_df= simOne[simOne['insulin'] != 0]
         edited_df.to_csv('./Results/{name}.csv', index=True, header=True)
        
def CreateCommonCSV():
    patients_df = pd.read_csv('./Results/patient100_updated.csv')
    patients_df = patients_df.drop(columns=['i'])

    columns = patients_df.columns.tolist()
    columns.append('insulin')
    columns.append('CGM')
    common_df = pd.DataFrame(columns=columns)
    for index, row in patients_df.iterrows():
            name = row['Name']
            row_without_name = row.drop('Name', errors='ignore')
            insulin_df = pd.read_csv(f'./Results/{name}.csv')
            first_insulin_value = insulin_df['insulin'].iloc[0]
            first_glucose_value = insulin_df['CGM'].iloc[0]
            row_without_name['insulin'] = first_insulin_value
            row_without_name['CGM'] = first_glucose_value
            common_df = common_df._append(row_without_name, ignore_index=True)
    common_df.to_csv('./Results/Common.csv')         
    
def TestPIDParams(dict): #adult_dic
    ResultParams = []
    for i in range(10):
        randP = random.uniform(0.00001,0.0001)
        randI = random.uniform(0.00001,0.0001)
        randD = random.uniform(0.00001,0.0001)
        ResultParams.append(str(i)+'. run   -P:'+str(randP)+'  I:'+str(randI)+'  D:'+str(randD))

        sim = Simulation('adult#001',dict,randP,randI,randD).getSimulationResults(1)
        edited_df= sim[sim['insulin'] != 0]
        edited_df.to_csv(f'./Results/['+str(i)+']PID_Test.csv', index=True, header=True)
        print(str(i)+'. Sim Done')
    
    for item in ResultParams:
        print(item)


def main():
    #CreateCommonCSV()
    adult_dic={'x0_ 1': 0.0,
               'x0_ 2': 0.0,
               'x0_ 3': 0.0,
               'x0_ 4': 265.370112,
               'x0_ 5': 162.457097269,
               'x0_ 6': 5.5043265,
               'x0_ 7': 0.0,
               'x0_ 8': 100.25,
               'x0_ 9': 100.25,
               'x0_10': 3.20762505142,
               'x0_11': 72.4341762342,
               'x0_12': 141.153779328,
               'x0_13': 265.370112,
                'BW': 102.32,
                'EGPb': 2.2758,
                'Gb': 138.56,
                'Ib': 100.25,
                'kabs': 0.08906,
                'kmax': 0.046122,
                'kmin': 0.0037927,
                'b': 0.70391,
                'd': 0.21057,
                'Vg': 1.9152,
                'Vi': 0.054906,
                'Ipb': 5.5043265,
                'Vmx': 0.031319,
                'Km0': 253.52,
                'k2': 0.087114,
                'k1': 0.058138, 'p2u': 0.027802, 'm1': 0.15446, 'm5': 0.027345, 'CL': 1.2642, 'HEb': 0.6, 'm2': 0.225027424083, 'm4': 0.090010969633, 'm30': 0.23169, 'Ilb': 3.20762505142, 'ki': 0.0046374, 'kp2': 0.00469, 'kp3': 0.01208, 'f': 0.9, 'Gpb': 265.370112, 'ke1': 0.0005, 'ke2': 339.0, 'Fsnc': 1.0, 'Gtb': 162.457097269, 'Vm0': 3.2667306607, 'Rdb': 2.2758, 'PCRb': 0.0164246535797, 'kd': 0.0152, 'ksc': 0.0766, 'ka1': 0.0019, 'ka2': 0.0078, 'dosekempt': 90000.0, 'u2ss': 1.2386244136, 'isc1ss': 72.4341762342, 'isc2ss': 141.153779328, 'kp1': 4.73140582528, 'patient_history': 0.0}
    
    TestPIDParams(adult_dic)

if __name__ == "__main__":
    main()