from datetime import date, datetime, timedelta
from simglucose.controller.base import Controller, Action
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
import numpy as np
from simglucose.simulation.user_interface import simulate

from Enviroment.deciisionTree import DecisionTree

class PDController(Controller):
    def __init__(self, P=1, D=0.1):
        self.P = P  # Proportional gain
        self.D = D  # Derivative gain
        self.prev_glucose = 0
        self.dose = 0
        self.min_glucose = 70
        self.max_glucose = 100
        self.iter = 0;
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
        self.P = 1
        self.D = 0.1  
        self.prev_error = 0
