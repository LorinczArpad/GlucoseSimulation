from datetime import datetime
from simglucose.controller.base import Controller, Action
from Enviroment.deciisionTree import DecisionTree

class PDController(Controller):
    def __init__(self):
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
            test_data = {
                'Time': [time],
                'CGM': [glucose],
                'LBGI': [lbgi],
                'HBGI': [hbgi],
                'Risk': [risk]
            }
            return Action(basal=self.decTree.create_result(test_data), bolus=0) 
                   
                
           
        
    def reset(self):
        self.decTree  = DecisionTree()
        self.decTree.train()
