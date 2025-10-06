from pathlib import Path
from PIL import ImageGrab
from colorama import Fore, Style, init as init_colorama
import imageio
import time
import numpy as np
import pandas as pd
import gymnasium
import random
from datetime import timedelta, datetime
from gymnasium.envs.registration import register
from simglucose.simulation.scenario import CustomScenario
from src.SimulationPreparation.SimulationConfig import SimulationConfig

from stable_baselines3 import A2C, TD3
from stable_baselines3.common.noise import NormalActionNoise
import logging
import pkg_resources
#import optuna
from stable_baselines3.common.evaluation import evaluate_policy
import ast
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env




class SimulationRunner:
    def __init__(self, env, lowmodel, innermodel, highmodel, config: SimulationConfig):
        self.env = env
        self.lowmodel = lowmodel
        self.innermodel = innermodel
        self.highmodel = highmodel
        self.config = config
        self.frames = []
        self.log_data = []
        self.insulin_timestamps = []

    def select_action(self, observation):
        obs_array = np.array([observation])
        if observation > 130:
            action, _ = self.highmodel.predict(obs_array, deterministic=True)
        elif 70 < observation <= 130:
            action, _ = self.innermodel.predict(obs_array, deterministic=True)
        else:
            action, _ = self.lowmodel.predict(obs_array, deterministic=True)
        return action

    def apply_insulin_rules(self, action, observation, risk, current_time):
        coefficient = 1.5 * risk if risk > 1 else 1
        if action > 0.1:
            action = 0.1 * coefficient
        if observation < 125:
            action = 0
        action = action * coefficient
        if action > 4:
            action = 3.5
        two_hour_ago = current_time - timedelta(hours=2)
        self.insulin_timestamps = [t for t in self.insulin_timestamps if t > two_hour_ago]
        if len(self.insulin_timestamps) >= 3:
            print(Fore.RED + f"Dosing prohibited! ({len(self.insulin_timestamps)} / 3)")
            print(Fore.RESET)
            action = 0
        else:
            if action > 0:
                self.insulin_timestamps.append(current_time)
                print(Fore.YELLOW + f"Insulin injected: {current_time.strftime('%H:%M')}")
                print(Fore.CYAN + f"Insulin injections in last 2 hours: {len(self.insulin_timestamps)} / 3")
                print(Fore.RESET)
        return action

    def run(self):
        logging.basicConfig(level=logging.INFO)
        observation, info = self.env.reset()
        current_time = self.config.start_time
        truncated = False
        risk = 0
        total_reward = 0
        step_count = 0
        tir_count = 0  # Count of steps within target glucose range (70–130)

        while current_time < self.config.start_time + timedelta(hours=24) and not truncated:
            self.env.render()
            current_time += timedelta(minutes=3)

            #Save and append single frame
            screen = ImageGrab.grab()
            frame = np.array(screen)
            self.frames.append(frame)

            action = self.select_action(observation[0])
            #action = self.apply_insulin_rules(action, observation[0], risk, current_time)
            observation, reward, terminated, truncated, info = self.env.step(action)
            risk = info["risk"]

            total_reward += reward
            step_count += 1
            if 70 <= observation[0] <= 130:
                tir_count += 1

            mean_reward = total_reward / step_count
            tir_percent = (tir_count / step_count) * 100

            #Log
             # Log data
            self.log_data.append({
                "action": action,
                "blood glucose": observation[0],
                "reward": reward,
                "mean_reward": mean_reward,
                "TIR (%)": tir_percent,
                "meal": info["meal"],
                "risk": info["risk"],
                "time": current_time
            })
            logging.info(
            f"Time: {current_time}, Action: {action}, BG: {observation[0]}, "
            f"Reward: {reward:.2f}, Mean Reward: {mean_reward:.2f}, TIR: {tir_percent:.1f}%"
        )
            
        return self.frames, self.log_data, truncated