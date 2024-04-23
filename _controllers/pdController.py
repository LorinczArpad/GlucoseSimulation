from datetime import datetime
from simglucose.controller.base import Controller, Action
from Enviroment.deciisionTree import DecisionTree
import pandas as pd
class PDController(Controller):
    def __init__(self, patientparams,P,I,D):
        self.patientparams = patientparams
        #PID CONTROLLER STUFF
        self.P =P
        self.I = I
        self.D = D
        self.target_glucose = 140
        self.prev_error = 0
        self.integral = 0
        #MACHINE LEARNING ALGO
        self.decTree  = DecisionTree()
        self.decTree.train()
    def compute_pid_control(self, current_glucose):
        error = self.target_glucose - current_glucose
        self.integral += error
        derivative = error - self.prev_error
        control = self.P * error + self.I * self.integral + self.D * derivative
        self.prev_error = error
        return control
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
            test_data.update({'CGM':obs.CGM})
            test_df = pd.DataFrame.from_dict(test_data, orient='index')
            test_df = pd.DataFrame([test_data], index=[0])
            print(f"BOLUS: {self.compute_pid_control(obs.CGM)}")
            return Action(basal=self.decTree.create_result(test_df), bolus=self.compute_pid_control(obs.CGM)) 
                   
                
           
        
    def reset(self):
        self.decTree  = DecisionTree()
        self.decTree.train()
        self.prev_error = 0
        self.integral = 0