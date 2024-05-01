from datetime import datetime
from simglucose.controller.base import Controller, Action
from Enviroment.deciisionTree import DecisionTree
from Enviroment.decisionTreeForPID import DecisionTreeForPID
import pandas as pd
import csv
class PDController(Controller):
    def __init__(self, patientparams):
        self.patientparams = patientparams
        #PID CONTROLLER STUFF
        self.P = 0
        self.I = 0
        self.D = 0
        self.target_glucose = 140
        self.prev_error = 0
        self.integral = 0
        #MACHINE LEARNING ALGO
        self.decTree  = DecisionTree()
        self.decTree.train()
        self.pid_Predicter = DecisionTreeForPID()
        self.pid_Predicter.train()
        
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
            
            pid_Parameters = self.pid_Predicter.create_PID_result(test_df)
            self.I = pid_Parameters['I']
            self.P = pid_Parameters['P']
            self.D = pid_Parameters['D']

            return Action(basal=self.decTree.create_result(test_df), bolus=self.compute_pid_control(obs.CGM)) 
                   
                
           
        
    def reset(self):
        self.decTree  = DecisionTree()
        self.decTree.train()
        self.prev_error = 0
        self.integral = 0