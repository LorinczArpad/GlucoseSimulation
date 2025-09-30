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

def generated_day(bw):
    from scipy import stats
    from random import randint
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

class HyperparameterTuner:
    def __init__(self, low_env, inner_env, high_env, n_trials=50, n_eval_episodes=5):
        self.low_env = low_env
        self.inner_env = inner_env
        self.high_env = high_env
        self.n_trials = n_trials
        self.n_eval_episodes = n_eval_episodes
        self.best_params = {
            "lowmodel": None,
            "innermodel": None,
            "highmodel": None
        }

    def objective(self, trial, env, model_name):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "n_steps": trial.suggest_int("n_steps", 5, 50),
            "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 2.0),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        }
        net_arch_options = [
            [64, 64],
            [128, 128],
            [256, 256]
        ]
        net_arch = net_arch_options[trial.suggest_categorical("net_arch", [0, 1, 2])]
        policy_kwargs = {"net_arch": net_arch}
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
            policy_kwargs=policy_kwargs,
            verbose=0,
            use_rms_prop=True,
            normalize_advantage=False,
        )
        try:
            model.learn(total_timesteps=500)
            mean_reward, std_reward = evaluate_policy(
                model,
                env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            trial.set_user_attr("mean_reward", mean_reward)
            trial.set_user_attr("std_reward", std_reward)
            trial.set_user_attr("params", params)
            trial.set_user_attr("net_arch", net_arch)
            return mean_reward
        except Exception as e:
            print(f"Trial failed for {model_name}: {e}")
            return -np.inf

    def tune_model(self, env, model_name):
        study = optuna.create_study(direction="maximize", study_name=f"a2c_{model_name}")
        study.optimize(
            lambda trial: self.objective(trial, env, model_name),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        best_trial = study.best_trial
        self.best_params[model_name] = {
            "params": best_trial.user_attrs["params"],
            "net_arch": best_trial.user_attrs["net_arch"],
            "mean_reward": best_trial.user_attrs["mean_reward"],
            "std_reward": best_trial.user_attrs["std_reward"],
            "value": best_trial.value
        }
        print(f"\nBest hyperparameters for {model_name}:")
        print(f"Parameters: {best_trial.user_attrs['params']}")
        print(f"Network Architecture: {best_trial.user_attrs['net_arch']}")
        print(f"Mean reward: {best_trial.user_attrs['mean_reward']:.2f} ± {best_trial.user_attrs['std_reward']:.2f}")
        return self.best_params[model_name]

    def tune_all(self):
        print("Tuning lowmodel...")
        self.tune_model(self.low_env, "lowmodel")
        print("\nTuning innermodel...")
        self.tune_model(self.inner_env, "innermodel")
        print("\nTuning highmodel...")
        self.tune_model(self.high_env, "highmodel")
        return self.best_params

    def save_results(self, filename="a2c_tuning_results.txt"):
        with open(filename, "w") as f:
            for model_name, result in self.best_params.items():
                if result is not None:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Best Parameters: {result['params']}\n")
                    f.write(f"Network Architecture: {result['net_arch']}\n")
                    f.write(f"Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}\n")
                    f.write(f"Best Value: {result['value']:.2f}\n")
                    f.write("-" * 50 + "\n")
        print(f"Results saved to {filename}")

class TD3HyperparameterTuner:
    def __init__(self, low_env, inner_env, high_env, n_trials=100, n_eval_episodes=10):
        self.low_env = low_env
        self.inner_env = inner_env
        self.high_env = high_env
        self.n_trials = n_trials
        self.n_eval_episodes = n_eval_episodes
        self.best_params = {
            "lowmodel": None,
            "innermodel": None,
            "highmodel": None
        }

    def objective(self, trial, env, model_name):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
            "buffer_size": trial.suggest_int("buffer_size", 10, 480, log=True),
            "learning_starts": trial.suggest_int("learning_starts", 100, 500),
            "batch_size": trial.suggest_int("batch_size", 16, 128, log=True),
            "tau": trial.suggest_float("tau", 5e-3, 5e-2, log=True),
            "gamma": trial.suggest_float("gamma", 0.95, 0.99),
            "train_freq": trial.suggest_int("train_freq", 1, 10),
            "gradient_steps": trial.suggest_int("gradient_steps", 1, 20),
            "policy_delay": trial.suggest_int("policy_delay", 2, 5),
            "target_policy_noise": trial.suggest_float("target_policy_noise", 0.05, 0.2),
            "target_noise_clip": trial.suggest_float("target_noise_clip", 0.1, 0.5),
        }
        action_noise_sigma = trial.suggest_float("action_noise_sigma", 0.1, 0.3, log=True)
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=action_noise_sigma * np.ones(n_actions)
        )
        net_arch_options = [
            [64, 64],
            [128, 128],
            [400, 300],
            dict(pi=[64, 64], qf=[256, 256])
        ]
        net_arch_idx = trial.suggest_categorical("net_arch", [0, 1, 2, 3])
        net_arch = net_arch_options[net_arch_idx]
        policy_kwargs = {"net_arch": net_arch}
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
            policy_kwargs=policy_kwargs,
            verbose=0,
            device="auto",
            seed=trial.number
        )
        try:
            model.learn(total_timesteps=500)
            mean_reward, std_reward = evaluate_policy(
                model,
                env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            trial.set_user_attr("mean_reward", mean_reward)
            trial.set_user_attr("std_reward", std_reward)
            trial.set_user_attr("params", params)
            trial.set_user_attr("net_arch", net_arch)
            trial.set_user_attr("action_noise_sigma", action_noise_sigma)
            return mean_reward
        except Exception as e:
            print(f"Trial failed for {model_name}: {e}")
            return -np.inf

    def tune_model(self, env, model_name):
        study = optuna.create_study(direction="maximize", study_name=f"td3_{model_name}")
        study.optimize(
            lambda trial: self.objective(trial, env, model_name),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        best_trial = study.best_trial
        self.best_params[model_name] = {
            "params": best_trial.user_attrs["params"],
            "net_arch": best_trial.user_attrs["net_arch"],
            "action_noise_sigma": best_trial.user_attrs["action_noise_sigma"],
            "mean_reward": best_trial.user_attrs["mean_reward"],
            "std_reward": best_trial.user_attrs["std_reward"],
            "value": best_trial.value
        }
        print(f"\nBest hyperparameters for {model_name}:")
        print(f"Parameters: {best_trial.user_attrs['params']}")
        print(f"Network Architecture: {best_trial.user_attrs['net_arch']}")
        print(f"Action Noise Sigma: {best_trial.user_attrs['action_noise_sigma']}")
        print(f"Mean reward: {best_trial.user_attrs['mean_reward']:.2f} ± {best_trial.user_attrs['std_reward']:.2f}")
        return self.best_params[model_name]

    def tune_all(self):
        print("Tuning lowmodel...")
        self.tune_model(self.low_env, "lowmodel")
        print("\nTuning innermodel...")
        self.tune_model(self.inner_env, "innermodel")
        print("\nTuning highmodel...")
        self.tune_model(self.high_env, "highmodel")
        return self.best_params

    def save_results(self, filename="td3_tuning_results.txt"):
        with open(filename, "w") as f:
            for model_name, result in self.best_params.items():
                if result is not None:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Best Parameters: {result['params']}\n")
                    f.write(f"Network Architecture: {result['net_arch']}\n")
                    f.write(f"Action Noise Sigma: {result['action_noise_sigma']}\n")
                    f.write(f"Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}\n")
                    f.write(f"Best Value: {result['value']:.2f}\n")
                    f.write("-" * 50 + "\n")
        print(f"Results saved to {filename}")
class PPOHyperparameterTuner:
    def __init__(self, low_env, inner_env, high_env, n_trials=50, n_eval_episodes=5):
        self.low_env = low_env
        self.inner_env = inner_env
        self.high_env = high_env
        self.n_trials = n_trials
        self.n_eval_episodes = n_eval_episodes
        self.best_params = {
            "lowmodel": None,
            "innermodel": None,
            "highmodel": None
        }

    def objective(self, trial, env, model_name):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_int("n_steps", 128, 2048, log=True),
            "batch_size": trial.suggest_int("batch_size", 32, 256, log=True),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "n_epochs": trial.suggest_int("n_epochs", 3, 20),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        }
        net_arch_options = [
            [64, 64],
            [128, 128],
            [256, 256],
            [400, 300]
        ]
        net_arch = net_arch_options[trial.suggest_categorical("net_arch", [0, 1, 2, 3])]
        policy_kwargs = {"net_arch": net_arch}

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            gae_lambda=params["gae_lambda"],
            gamma=params["gamma"],
            n_epochs=params["n_epochs"],
            ent_coef=params["ent_coef"],
            clip_range=params["clip_range"],
            policy_kwargs=policy_kwargs,
            verbose=0,
            device="auto",
            seed=trial.number
        )
        try:
            model.learn(total_timesteps=10000)  # Evaluate over 10,000 steps
            mean_reward, std_reward = evaluate_policy(
                model,
                env,  # Evaluate on single environment for consistency
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            trial.set_user_attr("mean_reward", mean_reward)
            trial.set_user_attr("std_reward", std_reward)
            trial.set_user_attr("params", params)
            trial.set_user_attr("net_arch", net_arch)
            return mean_reward
        except Exception as e:
            print(f"Trial failed for {model_name}: {e}")
            return -np.inf
        finally:
            env.close()

    def tune_model(self, env, model_name):
        study = optuna.create_study(direction="maximize", study_name=f"ppo_{model_name}")
        study.optimize(
            lambda trial: self.objective(trial, env, model_name),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        best_trial = study.best_trial
        self.best_params[model_name] = {
            "params": best_trial.user_attrs["params"],
            "net_arch": best_trial.user_attrs["net_arch"],
            "mean_reward": best_trial.user_attrs["mean_reward"],
            "std_reward": best_trial.user_attrs["std_reward"],
            "value": best_trial.value
        }
        print(f"\nBest hyperparameters for {model_name}:")
        print(f"Parameters: {best_trial.user_attrs['params']}")
        print(f"Network Architecture: {best_trial.user_attrs['net_arch']}")
        print(f"Mean reward: {best_trial.user_attrs['mean_reward']:.2f} ± {best_trial.user_attrs['std_reward']:.2f}")
        return self.best_params[model_name]

    def tune_all(self):
        print("Tuning lowmodel...")
        self.tune_model(self.low_env, "lowmodel")
        print("\nTuning innermodel...")
        self.tune_model(self.inner_env, "innermodel")
        print("\nTuning highmodel...")
        self.tune_model(self.high_env, "highmodel")
        return self.best_params

    def save_results(self, filename="ppo_tuning_results.txt"):
        with open(filename, "w") as f:
            for model_name, result in self.best_params.items():
                if result is not None:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Best Parameters: {result['params']}\n")
                    f.write(f"Network Architecture: {result['net_arch']}\n")
                    f.write(f"Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}\n")
                    f.write(f"Best Value: {result['value']:.2f}\n")
                    f.write("-" * 50 + "\n")
        print(f"Results saved to {filename}")
class SimulationConfig:
    def __init__(self, model_type="TD3"):
        self.save_to_csv = True
        self.save_video = True
        self.patient_name = "adult#002"
        self.start_time = datetime(2025, 1, 1, 0, 0, 0)
        self.time_steps = 100_000
        self.max_episode_steps = 480
        self.model_type = model_type
        self.model_name = model_type

    def get_patient_params(self):
        patient_params_file = pkg_resources.resource_filename("simglucose", "params/vpatient_params.csv")
        patient_params = pd.read_csv(patient_params_file)
        bw = patient_params[patient_params["Name"] == self.patient_name]["BW"].iloc[0]
        return {"bw": bw}

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
            ("simglucose/adolescent2-v0", "CustomT1DSimGymnaisumEnv"),
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

class ModelTrainer:
    def __init__(self, lowenv, innerenv, highenv, config: SimulationConfig):
        self.lowenv = lowenv
        self.innerenv = innerenv
        self.highenv = highenv
        self.config = config

    def load_params(self, filename):
        params_dict = {"lowmodel": None, "innermodel": None, "highmodel": None}
        errors = []
        
        if not Path(filename).exists():
            return None
       
        with open(filename, "r") as f:
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
                    params_dict[current_model] = {"params": {}}
                    
                elif line.startswith("Best Parameters:") and current_model:
                    params_str = line.split(": ", 1)[1] 
                    try:
                        params_dict[current_model]["params"] = ast.literal_eval(params_str)
                    except (SyntaxError, ValueError) as e:
                        errors.append(f"Error parsing Best Parameters for {current_model}: {e}")
                        params_dict[current_model] = None
                        
                elif line.startswith("Network Architecture:") and current_model:
                    net_arch_str = line.split(": ", 1)[1]
                    try:
                        params_dict[current_model]["net_arch"] = ast.literal_eval(net_arch_str)
                    except (SyntaxError, ValueError) as e:
                        errors.append(f"Error parsing Network Architecture for {current_model}: {e}")
                        params_dict[current_model] = None
                        
                elif line.startswith("Action Noise Sigma:") and current_model:
                    try:
                        params_dict[current_model]["action_noise_sigma"] = float(line.split(": ", 1)[1])
                    except ValueError as e:
                        errors.append(f"Error parsing Action Noise Sigma for {current_model}: {e}")
                        params_dict[current_model] = None
        
    
        for error in errors:
            print(error)
        
        if any(params_dict[model] for model in ["lowmodel", "innermodel", "highmodel"]):
            return params_dict
        
        print("Error: No valid parameters loaded for any model")
        return None
    

    def tune_models(self):
        filename = f"{self.config.model_type.lower()}_tuning_results.txt"
        best_params = self.load_params(filename)
  
        if best_params and all(best_params[model] for model in ["lowmodel", "innermodel", "highmodel"]):
            print(f"Loaded parameters from {filename}")
            return best_params
        if self.config.model_type == "A2C":
            tuner = HyperparameterTuner(self.lowenv, self.innerenv, self.highenv)
        if self.config.model_type == "PPO":
            tuner = PPOHyperparameterTuner(self.lowenv, self.innerenv, self.highenv)
        else:
            tuner = TD3HyperparameterTuner(self.lowenv, self.innerenv, self.highenv)
        best_params = tuner.tune_all()
        tuner.save_results(filename)
        return best_params

    def train_models(self, best_params):
     
        training_data_dir = Path(f"TrainingData/{self.config.patient_name}_{self.config.model_type}_Data")
    
        training_data_dir.mkdir(parents=True, exist_ok=True)
        
        models = {}
        for name, env in [("lowmodel", self.lowenv), ("innermodel", self.innerenv), ("highmodel", self.highenv)]:
            params = best_params[name]["params"]
            net_arch = best_params[name]["net_arch"]
            
            callback = RewardLoggerCallback()

            if self.config.model_type == "A2C":
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

            #Log
            self.log_data.append({
                "action": action,
                "blood glucose": observation[0],
                "reward": reward,
                "meal": info["meal"],
                "risk": info["risk"],
            })
            logging.info(f"Action taken: {action}, Blood Glucose: {observation[0]}, Reward: {reward}")
            
        return self.frames, self.log_data, truncated

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

def main():
    # Setup Config 
    config = SimulationConfig(model_type="PPO")
    patient_params = config.get_patient_params()
    print(f"Body weight for {config.patient_name}: {patient_params['bw']} kg")
    #Generate Meals
    meal_generator = MealGenerator(config)
    meal_scenario, meals = meal_generator.create_meal_scenario(patient_params["bw"])
    meal_generator.print_meals(meals)
    # Manage Enviroments
    env_manager = EnvironmentManager(config, meal_scenario)
    env_manager.register_environments()
    env, lowenv, innerenv, highenv = env_manager.create_environments()
    # Train Models 
    trainer = ModelTrainer(lowenv, innerenv, highenv, config)
    best_params = trainer.tune_models()
    lowmodel, innermodel, highmodel = trainer.train_models(best_params)
    # Run the simulation
    runner = SimulationRunner(env, lowmodel, innermodel, highmodel, config)
    frames, log_data, truncated = runner.run()
    # Save Result and metrics
    saver = DataSaver(env_manager.path_to_results, config)
    metrics_calculator = MetricsCalculator(env_manager.path_to_results, config)
    metrics = metrics_calculator.calculate_metrics(log_data)
    metrics_calculator.save_metrics(metrics)
    saver.save_video(frames)
    saver.save_csv(log_data)
    env.close() 

if __name__ == "__main__":
    main()