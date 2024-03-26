from datetime import datetime, timedelta
from simglucose.controller.base import Controller, Action
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
import numpy as np
from simglucose.simulation.user_interface import simulate

class PDController(Controller):
    def __init__(self, P=1, D=0.1):
        self.P = P  # Proportional gain
        self.D = D  # Derivative gain
        self.prev_glucose = 0
        self.dose = 0
        self.min_glucose = 70
        self.max_glucose = 100
        self.iter = 0;

    def policy(self, obs,reward,done, **info):

        return Action(basal=0.032, bolus=0) 
                   
                
           
        
    def reset(self):
        self.P = 1
        self.D = 0.1  
        self.prev_error = 0
