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
from datetime import datetime, timedelta

class CustomT1DSimGymnaisumEnv(T1DSimGymnaisumEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_time = datetime(2025, 1, 1, 0, 0, 0)#Szimuláció kezdő ideje éjfél

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        blood_glucose = observation[0]  
        target_bg = 120  
        bg_tolerance = 20         
        self.current_time += timedelta(minutes=3)
        deviation = abs(blood_glucose - target_bg)

        #00:00 és 06:00 közötti szigorúbb bünti       
        if self.current_time.hour < 6:
            if blood_glucose > 160:
                reward -=15
            elif 100 <= blood_glucose <=150:
                reward += 7
        
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

        
        print(f"[{self.current_time.strftime('%H:%M')}] Blood Glucose: {blood_glucose}, Reward: {reward}")

        return observation, reward, terminated, truncated, info
 
class LowGlucoseEnv(T1DSimGymnaisumEnv):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        blood_glucose = observation[0]
        
        # Target range: 70–180 mg/dL, with emphasis on avoiding hypoglycemia (<70)
        target_bg = 120  # Ideal glucose level
        bg_tolerance = 20  # Tolerance around target (100–140 mg/dL)
        
        # Base reward for staying in safe range
        if 70 <= blood_glucose <= 180:
            reward += 50 - 0.2 * (blood_glucose - target_bg) ** 2  # Quadratic reward, max at 120
        elif blood_glucose < 70:
            deviation = 70 - blood_glucose
            reward -= 20 * (deviation / 10)  # Linear penalty for hypoglycemia
        else:  # blood_glucose > 180
            deviation = blood_glucose - 180
            reward -= 10 * (deviation / 20)  # Mild penalty for hyperglycemia
        
        # Penalize high insulin during low glucose
        if action > 0.3 and blood_glucose < 70:
            reward -= 15  # Reduced penalty, only for significant insulin
        
        # Reward smooth glucose transitions
        if hasattr(self, 'last_blood_glucose'):
            fluctuation = abs(blood_glucose - self.last_blood_glucose)
            if fluctuation < 5:
                reward += 5  # Bonus for stability
            elif fluctuation > 15:
                reward -= 5  # Penalty for large swings
        
        self.last_blood_glucose = blood_glucose
        
        return observation, reward, terminated, truncated, info

class HighGlucoseEnv(T1DSimGymnaisumEnv):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        blood_glucose = observation[0]
        
        # Target range: 70–180 mg/dL, with emphasis on avoiding hyperglycemia (>180)
        target_bg = 120
        bg_tolerance = 20
        
        # Base reward for staying in safe range
        if 70 <= blood_glucose <= 180:
            reward += 50 - 0.2 * (blood_glucose - target_bg) ** 2  # Quadratic reward, max at 120
        elif blood_glucose > 180:
            deviation = blood_glucose - 180
            reward -= 20 * (deviation / 20)  # Linear penalty for hyperglycemia
        else:  # blood_glucose < 70
            deviation = 70 - blood_glucose
            reward -= 10 * (deviation / 10)  # Mild penalty for hypoglycemia
        
        # Penalize high insulin during low glucose
        if action > 0.3 and blood_glucose < 70:
            reward -= 15  # Reduced penalty, only for significant insulin
        
        # Reward smooth glucose transitions
        if hasattr(self, 'last_blood_glucose'):
            fluctuation = abs(blood_glucose - self.last_blood_glucose)
            if fluctuation < 5:
                reward += 5  # Bonus for stability
            elif fluctuation > 15:
                reward -= 5  # Penalty for large swings
        
        self.last_blood_glucose = blood_glucose
        
        return observation, reward, terminated, truncated, info

class InnerGlucoseEnv(T1DSimGymnaisumEnv):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        blood_glucose = observation[0]
        
        # Target range: 70–130 mg/dL, with emphasis on tight control
        target_bg = 100  # Tighter target for inner range
        bg_tolerance = 15  # Narrower tolerance (85–115 mg/dL)
        
        # Base reward for staying in tight range
        if 70 <= blood_glucose <= 130:
            reward += 60 - 0.3 * (blood_glucose - target_bg) ** 2  # Higher reward, max at 100
        elif blood_glucose < 70:
            deviation = 70 - blood_glucose
            reward -= 15 * (deviation / 10)  # Reduced penalty for hypoglycemia
        else:  # blood_glucose > 130
            deviation = blood_glucose - 130
            reward -= 10 * (deviation / 20)  # Mild penalty for exceeding 130
        
        # Penalize high insulin doses to encourage conservative dosing
        if action > 0.3:
            reward -= 5  # Mild penalty for high insulin
        
        # Reward smooth glucose transitions
        if hasattr(self, 'last_blood_glucose'):
            fluctuation = abs(blood_glucose - self.last_blood_glucose)
            if fluctuation < 5:
                reward += 5  # Bonus for stability
            elif fluctuation > 15:
                reward -= 5  # Penalty for large swings
        
        self.last_blood_glucose = blood_glucose
        
        return observation, reward, terminated, truncated, info
    