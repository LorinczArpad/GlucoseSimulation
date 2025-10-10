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
from src.SimulationPreparation.MealGenerator import MealGenerator
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


class EnvironmentManager:
    def __init__(self, config: SimulationConfig, meal_scenario):
        self.config = config
        self.base_kwargs = {
            "patient_name": config.patient_name,
            "custom_scenario": meal_scenario
        }
        self.path_to_results = self._create_results_directory()

    def _create_results_directory(self):
        base_folder = Path(f"SimResults/{self.config.model_name}_{self.config.patient_name}_00")
        counter = 0
        while base_folder.exists():
            suffix = f"_{counter:02d}"
            base_folder = Path(f"SimResults/{self.config.model_name}_{self.config.patient_name}{suffix}")
            counter += 1
        base_folder.mkdir(parents=True, exist_ok=False)
        print(f"Folder created: {base_folder.resolve()}")
        return base_folder

    def register_environments(self):
        env_configs = [
            ("simglucose/adolescent2-v0", "CustomT1DSimGymnasiumEnv"),
            ("simglucose/adolescent2-v0-low", "LowGlucoseEnv"),
            ("simglucose/adolescent2-v0-high", "HighGlucoseEnv"),
            ("simglucose/adolescent2-v0-inner", "InnerGlucoseEnv"),
        ]
        for env_id, entry_point in env_configs:
            register(
                id=env_id,
                entry_point=f"customEnviroments:{entry_point}",
                max_episode_steps=self.config.max_episode_steps,
                kwargs=self.base_kwargs,
            )

    def create_environments(self):
        lowenv = gymnasium.make("simglucose/adolescent2-v0-low", render_mode="human")
        innerenv = gymnasium.make("simglucose/adolescent2-v0-inner", render_mode="human")
        highenv = gymnasium.make("simglucose/adolescent2-v0-high", render_mode="human")
        env = gymnasium.make("simglucose/adolescent2-v0", render_mode="human")
        for e in [lowenv, innerenv, highenv]:
            e.reward_range = (-100, 100)
        return env, lowenv, innerenv, highenv