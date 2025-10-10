
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
import optuna
from stable_baselines3.common.evaluation import evaluate_policy
import ast
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env




class HyperparameterTuner:
    def __init__(self, low_env, inner_env, high_env, n_trials=10, n_eval_episodes=1):
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
            model.learn(total_timesteps=10000)
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
    def __init__(self, low_env, inner_env, high_env, n_trials=10, n_eval_episodes=1):
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
            model.learn(total_timesteps=10000)
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
    def __init__(self, low_env, inner_env, high_env, n_trials=10, n_eval_episodes=1):
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
            model.learn(total_timesteps=500)  # Evaluate over 10,000 steps
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
        study = optuna.create_study(direction="maximize", study_name=f"{model_name}")
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