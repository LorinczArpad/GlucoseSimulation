from datetime import datetime
from simglucose.controller.base import Controller, Action
from Enviroment.deciisionTree import DecisionTree
import pandas as pd
class PDController(Controller):
    def __init__(self, patientparams):
        self.patientparams = patientparams
        self.decTree  = DecisionTree()
        self.decTree.train()

    def policy(self, obs,reward,done, **info):
        if(done):
            return Action(basal=0, bolus=0) 
        else:
            
            lbgi = float(info['lbgi'])
            time = datetime.strptime(str(info['time']), '%Y-%m-%d %H:%M:%S.%f')
            # Convert 'hbgi' to float
            hbgi = float(info['hbgi'])
            glucose = obs.CGM
            # Convert 'risk' to float
            risk = float(info['risk'])
            test_data = self.patientparams
            test_df = pd.DataFrame.from_dict(test_data, orient='index')
            test_df = pd.DataFrame([test_data], index=[0])
            print(f'DATAFRAME{test_df}')
            return Action(basal=self.decTree.create_result(test_df), bolus=0) 
                   
                
           
        
    def reset(self):
        self.decTree  = DecisionTree()
        self.decTree.train()
