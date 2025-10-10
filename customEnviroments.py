from pathlib import Path
from PIL import ImageGrab
import imageio
import time
import numpy as np
import gymnasium
from gymnasium.envs.registration import register
from simglucose.simulation.scenario import CustomScenario
from datetime import datetime, timedelta
from collections import deque
from src.SimulationPreparation.MealGenerator import MealGenerator
from src.SimulationPreparation.SimulationConfig import SimulationConfig
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from simglucose.envs import T1DSimGymnaisumEnv

class BaseLookbackEnv(T1DSimGymnaisumEnv): 
    def __init__(self, *args, lookback=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookback_window = deque(maxlen=lookback)
        self.last_blood_glucose = None
        self.current_time = datetime(2025, 1, 1, 0, 0, 0)
        self.config = SimulationConfig(model_type="PPO")
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=self.MAX_BG, shape=(1,), dtype=np.float32
        )


    def reset(self, *, seed=None, options=None):

        patient_params = self.config.get_patient_params()
        bw = patient_params['bw']
        meal_generator = MealGenerator(self.config)
        meal_scenario, meals = meal_generator.create_meal_scenario(bw)
        self.custom_scenario = meal_scenario
        return  super().reset(seed=seed)


    def update_lookback(self, blood_glucose):
        self.lookback_window.append(blood_glucose)
        mean_bg = np.mean(self.lookback_window)
        std_bg = np.std(self.lookback_window)
        trend = blood_glucose - self.lookback_window[0] if len(self.lookback_window) > 1 else 0
        return mean_bg, std_bg, trend

    def _get_obs(self, bg):
        mean_bg, std_bg, trend = self.update_lookback(bg)
        hour = self.current_time.hour
        time_sin = np.sin(2 * np.pi * hour / 24)
        time_cos = np.cos(2 * np.pi * hour / 24)
        
        return np.array([bg, mean_bg, std_bg, trend, time_sin, time_cos], dtype=np.float32)

class CustomT1DSimGymnasiumEnv(BaseLookbackEnv): 
    def step(self, action):
        
        action = np.clip(action, 0, 0.5)
        observation, base_reward, terminated, truncated, info = super().step(action)
        bg = observation[0]  
        obs = self._get_obs(bg)  
        self.current_time += timedelta(minutes=5)  
        target_low, target_high = 70, 180
        is_in_range = 1 if target_low <= bg <= target_high else 0
        
        reward = 0  
        
        
        tir_weight = 10.0
        reward += tir_weight * is_in_range
        if not is_in_range:
            if bg < target_low:
                reward -= 15 * ((target_low - bg) / 20)  # Linear hypo
            else:
                reward -= 8 * ((bg - target_high) / 50)   # Softer hyper
        
        # 2. Correction Bonus
        if bg > target_high and action > 0.1:
            correction_scale = min(5, action * 2)
            reward += correction_scale
            info["correction_bonus"] = True
        
        # 3. Stability/Trend (20% mass)
        if len(self.lookback_window) > 1:
            if np.std(self.lookback_window) < 15:
                reward += 2
            if abs(self.lookback_window[-1] - self.lookback_window[0]) < 20:
                reward += 1
            else:
                reward -= 0.5
        
        # 4. Nighttime Safety
        if self.current_time.hour < 6:
            if bg > 150:
                reward -= 3
            elif bg < 90:
                reward -= 4
            else:
                reward += 5
        
        # Clip
        reward = np.clip(reward, -20, 15)
        
        self.last_blood_glucose = bg
        return np.array([observation[0]], dtype=np.float32), reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info

class LowGlucoseEnv(BaseLookbackEnv):
    def step(self, action):
        action = np.clip(action, 0, 0.5)
        observation, base_reward, terminated, truncated, info = super().step(action)
        bg = observation[0]
        obs = self._get_obs(bg)
        self.current_time += timedelta(minutes=5)
        
        # Reuse TIR logic, but bias toward hypo aversion
        target_low, target_high = 70, 180
        is_in_range = 1 if target_low <= bg <= target_high else 0
        
        reward = 0
        
        tir_weight = 10.0
        reward += tir_weight * is_in_range
        if bg < target_low:
            reward -= 20 * ((target_low - bg) / 20)  # Harsher hypo
        elif bg > target_high:
            reward -= 6 * ((bg - target_high) / 50)
        
        if bg > target_high and action > 0.1:
            reward += min(5, action * 2)
        
        if len(self.lookback_window) > 1:
            if np.std(self.lookback_window) < 15:
                reward += 2
            if abs(self.lookback_window[-1] - self.lookback_window[0]) < 20:
                reward += 1
            else:
                reward -= 0.5
        
        if self.current_time.hour < 6 and bg < 90:
            reward -= 5
        
        reward = np.clip(reward, -20, 15)
        self.last_blood_glucose = bg
        return np.array([observation[0]], dtype=np.float32), reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info

class HighGlucoseEnv(BaseLookbackEnv):
    def step(self, action):
        action = np.clip(action, 0, 0.5)
        observation, base_reward, terminated, truncated, info = super().step(action)
        bg = observation[0]
        obs = self._get_obs(bg)
        self.current_time += timedelta(minutes=5)
        
        target_low, target_high = 70, 180
        is_in_range = 1 if target_low <= bg <= target_high else 0
        
        reward = 0
        
        tir_weight = 10.0
        reward += tir_weight * is_in_range
        if bg < target_low:
            reward -= 12 * ((target_low - bg) / 20)
        elif bg > target_high:
            reward -= 10 * ((bg - target_high) / 50)  # Harsher hyper
        
        if bg > target_high and action > 0.1:
            reward += min(7, action * 3)  # Stronger correction bonus
        
        if len(self.lookback_window) > 1:
            if np.std(self.lookback_window) < 15:
                reward += 3
            if abs(self.lookback_window[-1] - self.lookback_window[0]) < 20:
                reward += 1
            else:
                reward -= 0.5
        
        reward = np.clip(reward, -20, 15)
        self.last_blood_glucose = bg
        return np.array([observation[0]], dtype=np.float32), reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info

class InnerGlucoseEnv(BaseLookbackEnv):
    def step(self, action):
        action = np.clip(action, 0, 0.5)
        observation, base_reward, terminated, truncated, info = super().step(action)
        bg = observation[0]
        obs = self._get_obs(bg)
        self.current_time += timedelta(minutes=5)
        
        target_low, target_high = 70, 180
        is_in_range = 1 if target_low <= bg <= target_high else 0
        
        reward = 0
        
        tir_weight = 10.0
        reward += tir_weight * is_in_range
        if bg < target_low:
            reward -= 15 * ((target_low - bg) / 20)
        elif bg > target_high:
            reward -= 8 * ((bg - target_high) / 50)
        
        if bg > target_high and action > 0.1:
            reward += min(4, action * 2)
        
        if len(self.lookback_window) > 1:
            if np.std(self.lookback_window) < 15:
                reward += 2
            if abs(self.lookback_window[-1] - self.lookback_window[0]) < 20:
                reward += 1
            else:
                reward -= 0.5
        
        if self.current_time.hour < 6:
            reward += 3 if 90 <= bg <= 140 else -2
        
        reward = np.clip(reward, -20, 15)
        self.last_blood_glucose = bg
        return np.array([observation[0]], dtype=np.float32), reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info