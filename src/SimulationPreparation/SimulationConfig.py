from pathlib import Path
from PIL import ImageGrab
from colorama import Fore, Style, init as init_colorama
import imageio
import time
import numpy as np
import pandas as pd
import gymnasium
import random
from datetime import timedelta, datetime as dt
from gymnasium.envs.registration import register
from simglucose.simulation.scenario import CustomScenario
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






class SimulationConfig:
    def __init__(self, model_type="TD3",low_model_type = "PPO",inner_model_type = "2AC",hight_model_type = "TD3"):
        self.save_to_csv = True
        self.save_video = True
        self.patient_name = "adult#002"
        self.start_time = dt(2025, 1, 1, 0, 0, 0)
        self.time_steps = 100
        self.max_episode_steps = 480
        self.model_type = model_type
        self.model_name = model_type
        self.low_model_type = low_model_type
        self.inner_model_type =inner_model_type
        self.high_model_tpye = hight_model_type

    def get_patient_params(self):
        patient_params_file = pkg_resources.resource_filename("simglucose", "params/vpatient_params.csv")
        patient_params = pd.read_csv(patient_params_file)
        bw = patient_params[patient_params["Name"] == self.patient_name]["BW"].iloc[0]
        return {"bw": bw}