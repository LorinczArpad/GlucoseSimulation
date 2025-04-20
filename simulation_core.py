import numpy as np
import pandas as pd
import os
import imageio
import gymnasium
import pkg_resources

from pathlib import Path
from datetime import datetime, timedelta
from PIL import ImageGrab
from random import randint
from colorama import Fore
from simglucose.simulation.scenario import CustomScenario
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


TIMESTEPS = 300
PATIENT_NAME = "adult#002"

# === Utility ===

def generated_day(bw):
    from scipy import stats
    meal = {
        "probability": [0.95, 0.3, 0.95, 0.3, 0.95, 0.3],
        "upperbound": [i * 60 for i in [9, 10, 14, 16, 20, 23]],
        "lowerbound": [i * 60 for i in [5, 9, 10, 14, 16, 20]],
        "upperboundmealtime": [30, 10, 30, 10, 30, 10],
        "loverboundmealtime": [10, 5, 10, 5, 10, 5],
        "meanmealtime": [20, 7.5, 20, 7, 20, 7.5],
        "meantime": [i * 60 for i in [7, 9.5, 12, 15, 18, 21.5]],
        "variancemealtime": [5, 2, 5, 2, 5, 2],
        "variancetime": [60, 30, 60, 30, 60, 30],
        "meanamount": [i * bw for i in [0.7, 0.15, 1.1, 0.15, 1.25, 0.15]],
        "varianceamount": [i * bw * 0.15 for i in [0.7, 0.15, 1.1, 0.15, 1.25, 0.15]],
        "E": []
    }
    for i in range(6):
        if randint(0, 100) / 100 < meal["probability"][i]:
            e = max(0, stats.norm(loc=meal["meanamount"][i], scale=meal["varianceamount"][i]).rvs())
            t = stats.truncnorm(
                (meal["lowerbound"][i] - meal["meantime"][i]) / meal["variancetime"][i],
                (meal["upperbound"][i] - meal["meantime"][i]) / meal["variancetime"][i],
                loc=meal["meantime"][i], scale=meal["variancetime"][i]).rvs()
            h = stats.truncnorm(
                (meal["loverboundmealtime"][i] - meal["meanmealtime"][i]) / meal["variancemealtime"][i],
                (meal["upperboundmealtime"][i] - meal["meanmealtime"][i]) / meal["variancemealtime"][i],
                loc=meal["meanmealtime"][i], scale=meal["variancemealtime"][i]).rvs()
            meal["E"].append([int(round(e)), int(round(t)), int(round(h))])
    return meal["E"]


def get_model_path(base_dir: Path, model_name: str) -> Path:
    return base_dir / f"{model_name}.zip"


def list_model_sets(root_dir: Path = Path("TrainingModels")):
    """
    Returns all directories inside root_dir that contain valid .zip model files
    """
    valid_sets = []
    if not root_dir.exists():
        return valid_sets

    for subdir in root_dir.iterdir():
        if subdir.is_dir():
            zip_files = list(subdir.glob("*.zip"))
            if any(m.stem in {"lowmodel", "innermodel", "highmodel"} for m in zip_files):
                valid_sets.append(subdir)

    return sorted(valid_sets, key=lambda p: p.stat().st_mtime, reverse=True)  # Most recent first


def prompt_user_to_choose_model_set():
    model_sets = list_model_sets()
    if not model_sets:
        print("No trained model sets found in 'TrainingModels/'.")
        return None

    most_recent = model_sets[0] if model_sets else None

    print("\nAvailable Trained Model Sets:")
    if most_recent:
        print(f" [0] Use most recently trained model ({most_recent.name})")
    print(" [1] Choose from list")
    print(" [2] Provide a custom path manually")

    try:
        choice = int(input("Select an option: "))
        if choice == 0 and most_recent:
            return most_recent
        elif choice == 1:
            print("\nModel Sets:")
            for i, path in enumerate(model_sets):
                print(f" [{i}] {path.name}")
            sub_choice = int(input("Choose model set: "))
            if 0 <= sub_choice < len(model_sets):
                return model_sets[sub_choice]
        elif choice == 2:
            custom_path = input("Enter full path to trained model directory: ").strip()
            path_obj = Path(custom_path)
            if path_obj.exists() and path_obj.is_dir():
                return path_obj
            else:
                print("Invalid path provided.")
    except ValueError:
        print("Invalid input. Training from scratch.")

    return None


def load_model_from_file(model_path: Path, model_type: str, env):
    model_class = A2C if model_type == "A2C" else TD3
    print(f"[Model I/O] Loading model from {model_path}")
    return model_class.load(str(model_path), env=env)


def save_model(model, base_dir: Path, model_name: str):
    model_path = get_model_path(base_dir, model_name)
    model.save(str(model_path))
    print(f"[Model I/O] Saved {model_name} model to {model_path}")


def clear_console():
    os.system('cls')


# === Config ===

class SimulationConfig:
    def __init__(self, model_type="TD3"):
        self.save_to_csv = True
        self.save_video = True
        self.render_sim = True
        self.patient_name = PATIENT_NAME
        self.start_time = datetime(2025, 1, 1, 0, 0, 0)
        self.time_steps = TIMESTEPS
        self.max_episode_steps = 480
        self.model_type = model_type
        self.model_name = model_type

    def get_patient_params(self):
        patient_params_file = pkg_resources.resource_filename("simglucose", "params/vpatient_params.csv")
        patient_params = pd.read_csv(patient_params_file)
        bw = patient_params[patient_params["Name"] == self.patient_name]["BW"].iloc[0]
        return {"bw": bw}

# === Scenario Generation ===

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

# === Environment Management ===

class EnvironmentManager:
    def __init__(self, config: SimulationConfig, meal_scenario):
        self.config = config
        self.base_kwargs = {
            "patient_name": config.patient_name,
            "custom_scenario": meal_scenario
        }
        self.path_to_results = self._create_results_directory()

    def _create_results_directory(self):
        base_folder = Path(f"SimResults/{self.config.model_name}_{self.config.patient_name}")
        counter = 0
        while base_folder.exists():
            counter += 1
            base_folder = Path(f"SimResults/{self.config.model_name}_{self.config.patient_name}_{counter:02d}")
        base_folder.mkdir(parents=True, exist_ok=False)
        print(f"Folder created: {base_folder.resolve()}")
        return base_folder

    def register_environments(self):
        envs = [
            ("simglucose/adolescent2-v0", "CustomT1DSimGymnaisumEnv"),
            ("simglucose/adolescent2-v0-low", "LowGlucoseEnv"),
            ("simglucose/adolescent2-v0-high", "HighGlucoseEnv"),
            ("simglucose/adolescent2-v0-inner", "InnerGlucoseEnv"),
        ]
        for env_id, entry_point in envs:
            register(id=env_id, entry_point=f"customEnviroments:{entry_point}",
                     max_episode_steps=self.config.max_episode_steps, kwargs=self.base_kwargs)

    def create_environments(self):
        render = "human" if self.config.render_sim else None
        env = gymnasium.make("simglucose/adolescent2-v0", render_mode=render)
        lowenv = gymnasium.make("simglucose/adolescent2-v0-low", render_mode=render)
        innerenv = gymnasium.make("simglucose/adolescent2-v0-inner", render_mode=render)
        highenv = gymnasium.make("simglucose/adolescent2-v0-high", render_mode=render)
        return env, lowenv, innerenv, highenv

# === Callback ===

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        self.rewards.append(self.locals['rewards'][0])
        return True

    def save_to_csv(self, filename):
        pd.DataFrame({'timestep': range(1, len(self.rewards)+1), 'reward': self.rewards}).to_csv(filename, index=False)

# === Trainer ===

class ModelTrainer:
    def __init__(self, lowenv, innerenv, highenv, config: SimulationConfig):
        self.envs = {"lowmodel": lowenv, "innermodel": innerenv, "highmodel": highenv}
        self.config = config
        self.models = {}

    def train_or_load_models(self, use_existing_models=False):
        if use_existing_models:
            base_dir = prompt_user_to_choose_model_set()
            if base_dir is None:
                print("No model set selected. Training from scratch.")
                use_existing_models = False
        else:
            base_dir = Path(f"TrainingModels/{self.config.patient_name}_{self.config.model_type}_00")
            counter = 0
            while base_dir.exists():
                counter += 1
                base_dir = Path(f"TrainingModels/{self.config.model_name}_{self.config.patient_name}_{counter:02d}")
            base_dir.mkdir(parents=True, exist_ok=False)

        for model_name, env in self.envs.items():
            callback = RewardLoggerCallback()
            model = None

            if use_existing_models:
                model_path = get_model_path(base_dir, model_name)
                if model_path.exists():
                    model = load_model_from_file(model_path, self.config.model_type, env)
                    print(f"[{model_name}] loaded from {model_path}")
                else:
                    print(f"[{model_name}] not found in {base_dir}. Will train from scratch.")

            if model is None:
                print(f"Training new model: {model_name}")
                if self.config.model_type == "A2C":
                    model = A2C("MlpPolicy", env, verbose=1)
                else:
                    action_noise = NormalActionNoise(
                        mean=np.zeros(env.action_space.shape[-1]),
                        sigma=0.1 * np.ones(env.action_space.shape[-1])
                    )
                    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)

                model.learn(total_timesteps=self.config.time_steps, callback=callback)
                if not use_existing_models:
                    save_model(model, base_dir, model_name)
                    callback.save_to_csv(base_dir / f"{model_name}_rewards.csv")

            self.models[model_name] = model

        clear_console()
        return self.models["lowmodel"], self.models["innermodel"], self.models["highmodel"]

# === Simulation Runner ===

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

    def select_action(self, obs):
        value = obs[0]
        obs_array = np.array([obs])
        if value > 130:
            action, _ = self.highmodel.predict(obs_array, deterministic=True)
        elif 70 < value <= 130:
            action, _ = self.innermodel.predict(obs_array, deterministic=True)
        else:
            action, _ = self.lowmodel.predict(obs_array, deterministic=True)
        return action

    def apply_insulin_rules(self, action, observation, risk, current_time):
        coefficient = 1.5 * risk if risk > 1 else 1
        action = min(action, 0.1) * coefficient
        if observation < 125:
            action = 0
        action = min(action, 3.5)

        # Dosing limits
        self.insulin_timestamps = [t for t in self.insulin_timestamps if t > current_time - timedelta(hours=2)]
        if len(self.insulin_timestamps) >= 3:
            print(Fore.RED + f"[Dosing Prohibited] Too many injections in last 2 hrs.")
            action = 0
        elif action > 0:
            self.insulin_timestamps.append(current_time)
            print(Fore.YELLOW + f"Injected insulin at {current_time.strftime('%H:%M')}")

        return action

    def run(self):
        obs, info = self.env.reset()
        risk = 0
        current_time = self.config.start_time
        end_time = current_time + timedelta(hours=24)
        truncated = False

        while current_time < end_time and not truncated:
            if self.config.render_sim:
                self.env.render()
            if self.config.save_video:
                screen = ImageGrab.grab()
                self.frames.append(np.array(screen))

            current_time += timedelta(minutes=3)
            action = self.select_action(obs)
            action = self.apply_insulin_rules(action, obs[0], risk, current_time)
            obs, reward, terminated, truncated, info = self.env.step(action)
            risk = info.get("risk", 0)

            self.log_data.append({
                "action": action,
                "blood glucose": obs[0],
                "reward": reward,
                "meal": info.get("meal", 0),
                "risk": risk,
                "time": current_time.strftime("%H:%M")
            })

        return self.frames, self.log_data

# === Data Saving ===

class DataSaver:
    def __init__(self, path: Path, config: SimulationConfig):
        self.path = path
        self.config = config

    def save_csv(self, data, filename="LogData.csv"):
        if self.config.save_to_csv:
            df = pd.DataFrame(data)
            df.to_csv(self.path / filename, index=False)
            print(Fore.GREEN + f"Saved CSV: {self.path / filename}")

    def save_video(self, frames, filename="Simulation.mp4"):
        if self.config.save_video:
            print(Fore.YELLOW + "Saving video... this may take a moment.")
            imageio.mimsave(self.path / filename, frames, fps=20)
            print(Fore.GREEN + f"Saved video: {self.path / filename}")

# === Metrics ===

class MetricsCalculator:
    def __init__(self, path: Path):
        self.path = path

    def calculate(self, log_data):
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

    def save(self, metrics, filename="metrics.txt"):
        with open(self.path / filename, "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.2f}\n")
        print(Fore.GREEN + f"Saved metrics: {self.path / filename}")
