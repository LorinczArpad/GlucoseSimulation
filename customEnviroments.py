from pathlib import Path
from PIL import ImageGrab
import imageio
import time
import numpy as np
import gymnasium
from gymnasium.envs.registration import register
from simglucose.simulation.scenario import CustomScenario
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from simglucose.envs import T1DSimGymnaisumEnv

class CustomT1DSimGymnaisumEnv(T1DSimGymnaisumEnv):
    def step(self, action):
        
        observation, reward, terminated, truncated, info = super().step(action)
        blood_glucose = observation[0]  
        target_bg = 120  
        bg_tolerance = 20  

        
        deviation = abs(blood_glucose - target_bg)

        
        if blood_glucose > target_bg + bg_tolerance:
           
            reward -= 5 * (deviation / 10) 
        elif blood_glucose < target_bg - bg_tolerance:
           
            reward -= 5 * (deviation / 10) 
        else:
            
            reward += 5 

            if hasattr(self, 'last_blood_glucose'):
                fluctuation = abs(blood_glucose - self.last_blood_glucose)
                if fluctuation < 10: 
                    reward += 2  
                elif fluctuation >= 10 and fluctuation < 20:  
                    reward -= 1  
            
          
            self.last_blood_glucose = blood_glucose

        return observation, reward, terminated, truncated, info 
    
class LowGlucoseEnv(T1DSimGymnaisumEnv):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        blood_glucose = observation[0]
        
      
        if blood_glucose < 70:
            reward -= 10  
        elif blood_glucose < 90:
            reward -= 5  
        else:
            reward += 5  

        self.last_blood_glucose = blood_glucose
        return observation, reward, terminated, truncated, info


class HighGlucoseEnv(T1DSimGymnaisumEnv):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        blood_glucose = observation[0]
        
       
        if blood_glucose > 180:
            reward -= 10  
        elif blood_glucose > 140:
            reward -= 5  
        else:
            reward += 5  

        self.last_blood_glucose = blood_glucose
        return observation, reward, terminated, truncated, info


class InnerGlucoseEnv(T1DSimGymnaisumEnv):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        blood_glucose = observation[0]
        
        
        if blood_glucose < 70:
            reward -= 10  
        elif blood_glucose > 130:
            reward -= 10  
        elif 70 <= blood_glucose <= 130:
            reward += 5  

        return observation, reward, terminated, truncated, info
   