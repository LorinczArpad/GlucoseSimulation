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
from src.HyperParameterTuning.HyperParameterTuner import HyperparameterTuner, PPOHyperparameterTuner, TD3HyperparameterTuner
from src.SimulationPreparation import SimulationConfig
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




class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.rewards = []  

    def _on_step(self) -> bool:
        self.rewards.append(self.locals['rewards'][0])  
        return True

    def save_to_csv(self, filename):
        df = pd.DataFrame({
            'timestep': range(1, len(self.rewards) + 1),
            'reward': self.rewards
        })
        df.to_csv(filename, index=False)


class ModelTrainer:
    def __init__(self, lowenv, innerenv, highenv, config: SimulationConfig):
        self.lowenv = lowenv
        self.innerenv = innerenv
        self.highenv = highenv
        self.config = config

    def load_params(self):
        # Fixed filenames
        filenames = {
            "lowmodel": "low_tuning_results.txt",
            "innermodel": "inner_tuning_results.txt",
            "highmodel": "high_tuning_results.txt"
        }

        # Initialize dictionary
        params_dict = {"lowmodel": None, "innermodel": None, "highmodel": None}
        errors = []

        for model_name, fname in filenames.items():
            if not Path(fname).exists():
                continue

            with open(fname, "r") as f:
                content = f.read()

            sections = content.split("--------------------------------------------------")

            for section in sections:
                lines = section.strip().split("\n")
                current_model = None

                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("-"):
                        continue

                    if line.startswith("Model:"):
                        current_model = line.split(": ", 1)[1]
                        if params_dict[model_name] is None:
                            params_dict[model_name] = {"params": {}}

                    elif line.startswith("Best Parameters:") and current_model:
                        params_str = line.split(": ", 1)[1]
                        try:
                            params_dict[model_name]["params"] = ast.literal_eval(params_str)
                        except (SyntaxError, ValueError) as e:
                            errors.append(f"Error parsing Best Parameters for {model_name}: {e}")
                            params_dict[model_name] = None

                    elif line.startswith("Network Architecture:") and current_model:
                        net_arch_str = line.split(": ", 1)[1]
                        try:
                            params_dict[model_name]["net_arch"] = ast.literal_eval(net_arch_str)
                        except (SyntaxError, ValueError) as e:
                            errors.append(f"Error parsing Network Architecture for {model_name}: {e}")
                            params_dict[model_name] = None

                    elif line.startswith("Action Noise Sigma:") and current_model:
                        try:
                            params_dict[model_name]["action_noise_sigma"] = float(line.split(": ", 1)[1])
                        except ValueError as e:
                            errors.append(f"Error parsing Action Noise Sigma for {model_name}: {e}")
                            params_dict[model_name] = None

        # Print errors
        for error in errors:
            print(error)

        return params_dict
    
    def tune_models(self):
        filename = f"{self.config.model_type.lower()}_tuning_results.txt"
        best_params = self.load_params()
  
        if best_params and all(best_params[model] for model in ["lowmodel", "innermodel", "highmodel"]):
            print(f"Loaded parameters from {filename}")
            return best_params
        if self.config.low_model_type == "A2C":
            low_tuner = HyperparameterTuner(self.lowenv,self.innerenv,self.highenv)
        elif self.config.low_model_type == "PPO":
            low_tuner = PPOHyperparameterTuner(self.lowenv,self.innerenv,self.highenv)
        else:
            low_tuner = TD3HyperparameterTuner(self.lowenv,self.innerenv,self.highenv)
        best_params["lowmodel"] = low_tuner.tune_model(self.lowenv,"lowmodel")

        # --- Inner Environment Tuning ---
        if self.config.inner_model_type == "A2C":
            inner_tuner = HyperparameterTuner(self.lowenv,self.innerenv,self.highenv)
        elif self.config.inner_model_type == "PPO":
            inner_tuner = PPOHyperparameterTuner(self.lowenv,self.innerenv,self.highenv)
        else:
            inner_tuner = TD3HyperparameterTuner(self.lowenv,self.innerenv,self.highenv)
        best_params["innermodel"] = inner_tuner.tune_model(self.innerenv,"innermodel")

        # --- High Environment Tuning ---
        if self.config.high_model_tpye == "A2C":
            high_tuner = HyperparameterTuner(self.lowenv,self.innerenv,self.highenv)
        elif self.config.high_model_tpye == "PPO":
            high_tuner = PPOHyperparameterTuner(self.lowenv,self.innerenv,self.highenv)
        else:
            high_tuner = TD3HyperparameterTuner(self.lowenv,self.innerenv,self.highenv)
        best_params["highmodel"] = high_tuner.tune_model(self.highenv,"highmodel")
        #Dont change filename it is needed for load params
        low_tuner.save_results("low" + "_tuning_results.txt")
        inner_tuner.save_results("inner" + "_tuning_results.txt")
        high_tuner.save_results("high "+ "_tuning_results.txt")
        return best_params

    def train_models(self, best_params):
        training_data_dir = Path(f"TrainingData/{self.config.patient_name}_{self.config.model_type}_Data")
        training_data_dir.mkdir(parents=True, exist_ok=True)

        models = {}

        model_configs = [
            ("lowmodel", self.lowenv, self.config.low_model_type),
            ("innermodel", self.innerenv, self.config.inner_model_type),
            ("highmodel", self.highenv, self.config.high_model_tpye)
        ]

        for name, env, model_type in model_configs:
            params = best_params[name]["params"]
            net_arch = best_params[name]["net_arch"]
            callback = RewardLoggerCallback()

            # --- A2C ---
            if model_type == "A2C":
                model = A2C(
                    policy="MlpPolicy",
                    env=env,
                    learning_rate=params["learning_rate"],
                    n_steps=params["n_steps"],
                    vf_coef=params["vf_coef"],
                    ent_coef=params["ent_coef"],
                    max_grad_norm=params["max_grad_norm"],
                    gae_lambda=params["gae_lambda"],
                    gamma=params["gamma"],
                    policy_kwargs={"net_arch": net_arch},
                    verbose=1
                )

            # --- PPO ---
            elif model_type == "PPO":
                model = PPO(
                    policy="MlpPolicy",
                    env=env,
                    learning_rate=params["learning_rate"],
                    n_steps=params["n_steps"],
                    batch_size=params["batch_size"],
                    n_epochs=params["n_epochs"],
                    gamma=params["gamma"],
                    gae_lambda=params["gae_lambda"],
                    clip_range=params["clip_range"],
                    ent_coef=params["ent_coef"],
                    vf_coef=params["vf_coef"],
                    max_grad_norm=params["max_grad_norm"],
                    policy_kwargs={"net_arch": net_arch},
                    verbose=1
                )

            # --- TD3 ---
            else:
                action_noise = NormalActionNoise(
                    mean=np.zeros(env.action_space.shape[-1]),
                    sigma=best_params[name]["action_noise_sigma"] * np.ones(env.action_space.shape[-1])
                )

                model = TD3(
                    policy="MlpPolicy",
                    env=env,
                    learning_rate=params["learning_rate"],
                    buffer_size=params["buffer_size"],
                    learning_starts=params["learning_starts"],
                    batch_size=params["batch_size"],
                    tau=params["tau"],
                    gamma=params["gamma"],
                    train_freq=params["train_freq"],
                    gradient_steps=params["gradient_steps"],
                    action_noise=action_noise,
                    policy_delay=params["policy_delay"],
                    target_policy_noise=params["target_policy_noise"],
                    target_noise_clip=params["target_noise_clip"],
                    policy_kwargs={"net_arch": net_arch},
                    verbose=1
                )

            print(f"\n--- Training {name} ({model_type}) ---\n")
            model.learn(total_timesteps=self.config.time_steps, callback=callback)
            callback.save_to_csv(training_data_dir / f"{name}_rewards.csv")
            models[name] = model

        self.plot_and_save_rewards(training_data_dir)

        return models["lowmodel"], models["innermodel"], models["highmodel"]


    def plot_and_save_rewards(self, training_data_dir):
        model_names = ["lowmodel", "innermodel", "highmodel"]
        for model in model_names:
            csv_file = training_data_dir / f"{model}_rewards.csv"
            if not csv_file.exists():
                print(f"Warning: {csv_file} not found. Skipping.")
                continue
            df = pd.read_csv(csv_file)
            if df.empty:
                print(f"Warning: {csv_file} is empty. Skipping.")
                continue
            
            
            total_timesteps = df['timestep'].max()
            
            
            if total_timesteps >= 3_000_000:
                interval = 30_000  
            elif total_timesteps >= 1_000_000:
                interval = 10_000  
            elif total_timesteps >= 500_000:
                interval = 5_000   
            elif total_timesteps >= 100_000:
                interval = 1_000   
            elif total_timesteps >= 50_000:
                interval = 500     
            elif total_timesteps >= 10_000:
                interval = 100     
            elif total_timesteps >= 1_000:
                interval = 10      
            else:
                interval = 1
            
            if interval > 1:
               
                df['group'] = (df['timestep'] - 1) // interval
                grouped = df.groupby('group').agg({'timestep': 'min', 'reward': 'mean'})
                x = grouped['timestep']  
                y = grouped['reward']    
                label = f"{model} Reward (avg every {interval} steps)"
            else:
               
                x = df['timestep']
                y = df['reward']
                label = f"{model} Reward"
            
            
            plt.figure()
            plt.plot(x, y, label=label)
            plt.xlabel('Timestep')
            plt.ylabel('Reward')
            plt.title(f'{model.capitalize()} Rewards Over Time')
            plt.legend()
            plt.savefig(training_data_dir / f"{model}_rewards.png")
            plt.close()
            print(f"Plot saved: {training_data_dir / f'{model}_rewards.png'}")


class DataSaver:
    def __init__(self, path: Path, config: SimulationConfig):
        self.path = path
        self.config = config

    def save_csv(self, data, filename="LogData.csv"):
        if self.config.save_to_csv:
            init_colorama(True)
            full_path = self.path / filename
            df = pd.DataFrame(data)
            df.to_csv(full_path, index=False)
            print(Fore.GREEN + f"Data saved at {full_path}")

    def save_video(self, frames, filename="PlotVideo.mp4"):
        if self.config.save_video:
            init_colorama(True)
            full_path = self.path / filename
            print(Fore.RED + "Saving video, this may take several seconds or minutes!")
            imageio.mimsave(full_path, frames, format="mp4", fps=20)
            print(Fore.GREEN + f"Video saved at {full_path}")

class MetricsCalculator:
    def __init__(self, path: Path, config: SimulationConfig):
        self.path = path
        self.config = config

    def calculate_metrics(self, log_data):
        df = pd.DataFrame(log_data)
        tir = ((df["blood glucose"] >= 70) & (df["blood glucose"] <= 130)).mean() * 100
        hypo = (df["blood glucose"] < 70).sum()
        hyper = (df["blood glucose"] > 180).sum()
        mean_risk = df["risk"].mean()
        avg_reward = df["reward"].mean()
        return {
            "TIR (%)": tir,
            "Hypo Events": hypo,
            "Hyper Events": hyper,
            "Mean Risk": mean_risk,
            "Average Reward": avg_reward
        }

    def save_metrics(self, metrics, filename="metrics.txt"):
        full_path = self.path / filename
        with open(full_path, "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value:.2f}\n")
        print(Fore.GREEN + f"Metrics saved at {full_path}")