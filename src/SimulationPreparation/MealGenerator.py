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
from scipy import stats
from random import randint

def generated_day(bw):

    meal = {
        "probability": [0.95, 0.3, 0.95, 0.3, 0.95, 0.3],
        "upperbound": [i * 60 for i in [9, 10, 14, 16, 20, 23]],
        "lowerbound": [i * 60 for i in [5, 9, 10, 14, 16, 20]],
        "upperboundmealtime": [i for i in [30, 10, 30, 10, 30, 10]],
        "loverboundmealtime": [i for i in [10, 5, 10, 5, 10, 5]],
        "meanmealtime": [i for i in [20, 7.5, 20, 7, 0.5, 20, 7.5]],
        "meantime": [i * 60 for i in [7, 9.5, 12, 15, 18, 21.5]],
        "variancemealtime": [i for i in [5, 2, 5, 2, 5, 2]],
        "variancetime": [60, 30, 60, 30, 60, 30],
        "meanamount": [i * bw for i in [0.7, 0.15, 1.1, 0.15, 1.25, 0.15]],
        "varianceamount": [i * bw * 0.15 for i in [0.7, 0.15, 1.1, 0.15, 1.25, 0.15]],
        "E": []
    }
    for i in range(0, 6):
        value = randint(0, 100)
        if value / 100 < meal["probability"][i]:
            s = stats.norm(loc=meal["meanamount"][i], scale=meal["varianceamount"][i])
            s = s.rvs(10000)[randint(0, 9999)]
            e = round(max(0, s))
            s = stats.truncnorm(
                (meal["lowerbound"][i] - meal["meantime"][i]) / meal["variancetime"][i],
                (meal["upperbound"][i] - meal["meantime"][i]) / meal["variancetime"][i],
                loc=meal["meantime"][i], scale=meal["variancetime"][i]
            )
            s = s.rvs(10000)[randint(0, 9999)]
            t = round(s)
            s = stats.truncnorm(
                (meal["loverboundmealtime"][i] - meal["meanmealtime"][i]) / meal["variancemealtime"][i],
                (meal["upperboundmealtime"][i] - meal["meanmealtime"][i]) / meal["variancemealtime"][i],
                loc=meal["meanmealtime"][i], scale=meal["variancemealtime"][i]
            )
            s = s.rvs(10000)[randint(0, 9999)]
            h = round(s)
            meal["E"].append([int(e), int(t), int(h)])
    return meal["E"]

class MealGenerator:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def create_meal_scenario(self, bw):
        meal_events = generated_day(bw)
        meals = [(event[2], event[0]) for event in meal_events]
        return CustomScenario(start_time=self.config.start_time, scenario=meals), meals

    def print_meals(self, meals):
        print("CHO intakes:")
        for meal in meals:
            print(Fore.BLUE + f"   Meal: at {meal[0]} o'clock {meal[1]}g")
        print(Fore.RESET)