from datetime import datetime
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.sim_engine import SimObj, sim
from datetime import datetime, timedelta
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController

from _controllers.pdController import PDController


class Simulation:
    def __init__(self,patientName:str,patientParams):
        #Conroller
        self.controller = PDController(patientParams,0.001,0.0001,0.00001)
        #BBController()
        #PDController()
        # Sensor
        self.sensor = CGMSensor.withName('Dexcom',seed=1)
        
        # Pump
        self.pump = InsulinPump.withName('Insulet')

        # Patient
        self.patient = T1DPatient.withName(patientName)
        # Simulation Scenario
        start_time = datetime.now()
        self.scenario = RandomScenario(start_time=start_time, seed=1)
        
    def getSimulationObject(self, numberOfDays):
        env = T1DSimEnv(self.patient, self.sensor, self.pump, self.scenario)
        return SimObj(env,self.controller,timedelta(days=numberOfDays),animate=False, path='./Results')
    
    def getSimulationResults(self, numberOfDays):
        return sim(self.getSimulationObject(numberOfDays))