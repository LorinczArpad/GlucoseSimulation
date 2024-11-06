from pathlib import Path
from PIL import ImageGrab
from colorama import Fore, Style, init as init_colorama
import imageio
import time
import numpy as np
import pandas as pd
import gymnasium
from gymnasium.envs.registration import register
from simglucose.simulation.scenario import CustomScenario
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from simglucose.envs import T1DSimGymnaisumEnv
import logging
from simglucose.simulation.scenario import CustomScenario

def create_and_get_directory_for_sim_results(kwargs):
    #Create folder with patient name
    patient_name = kwargs.get("patient_name")
    base_folder = Path("SimResults/" + str(patient_name) + "_00")
    counter = 0
    while base_folder.exists():
        suffix = f"_{counter:02d}"
        base_folder = Path(f"SimResults/{patient_name}{suffix}")
        counter += 1

    base_folder.mkdir(parents=True, exist_ok=False)
    return base_folder.resolve()

def save_data_as_csv(save_csv : bool, path, data, filename):
    if(save_csv):
        init_colorama(True)
        full_path = str(path) + "/" + str(filename)
        df = pd.DataFrame(data)
        df.to_csv(full_path, index=False)
        print(Fore.GREEN + f"Data saved at {full_path}")

def save_plot_as_video(save_plot : bool, path, frames, filename):
    if(save_plot):
        init_colorama(True)
        full_path = str(path) + "/" + str(filename)
        print(Fore.RED + "Saving the video, THIS WILL TAKE TIME, AT LEAST SEVERAL SECONDS OR EVEN A MINUTE!")
        imageio.mimsave(full_path, frames, format="mp4", fps=20)  # Save frames as video
        print(Fore.GREEN + f"Video saved at {full_path}")

def main():
    SAVE_TO_CSV = True
    SAVE_VIDEO = True

    meal_events = [
        (8, 25), 
        (13, 20),   
        (19, 15),   
    ]

    start_time = datetime(2025, 1, 1, 6, 0, 0)

    meal_scenario = CustomScenario(start_time=start_time, scenario=meal_events)

    base_kwargs = {"patient_name": "adolescent#002", 'custom_scenario': meal_scenario}

    path_to_SimResults_sub_folders = create_and_get_directory_for_sim_results(base_kwargs)

    print(f"Folder created: {path_to_SimResults_sub_folders}")
    
    register(
        id="simglucose/adolescent2-v0",
        entry_point="customEnviroments:CustomT1DSimGymnaisumEnv",
        max_episode_steps=1000,
        kwargs=base_kwargs,
    )
    register(
        id="simglucose/adolescent2-v0-low",
        entry_point="customEnviroments:LowGlucoseEnv",
        max_episode_steps=1000,
        kwargs=base_kwargs,
    )
    register(
        id="simglucose/adolescent2-v0-high",
        entry_point="customEnviroments:HighGlucoseEnv",
        max_episode_steps=1000,
        kwargs=base_kwargs,
    )
    register(
        id="simglucose/adolescent2-v0-inner",
        entry_point="customEnviroments:InnerGlucoseEnv",
        max_episode_steps=1000,
        kwargs=base_kwargs,
    )

    #Envs
    lowenv = gymnasium.make("simglucose/adolescent2-v0-low", render_mode="human")  # Low BG
    innerenv = gymnasium.make("simglucose/adolescent2-v0-inner", render_mode="human")  # Inner BG
    highenv = gymnasium.make("simglucose/adolescent2-v0-high", render_mode="human")  # High BG
    #Reward Ranges
    lowenv.reward_range = (-100,100)
    innerenv.reward_range = (-100,100)
    highenv.reward_range = (-100,100)

    env = gymnasium.make("simglucose/adolescent2-v0", render_mode="human")  # Controll Env
    #Models
    lowmodel = PPO("MlpPolicy", lowenv, verbose=1)
    innermodel = PPO("MlpPolicy", innerenv, verbose=1)
    highmodel = PPO("MlpPolicy", highenv, verbose=1)
    #Training
    lowmodel.learn(total_timesteps=500)
    innermodel.learn(total_timesteps=500)
    highmodel.learn(total_timesteps=500)
    #model = PPO("MlpPolicy", env, verbose=1)
    #model.learn(total_timesteps=500)

    logging.basicConfig(level=logging.INFO)

    frames = []
    log_data = []

    observation, info = env.reset()
    truncated = False
    while(truncated != True):
        env.render()

        # Capture screen
        screen = ImageGrab.grab()
        frame = np.array(screen)
        frames.append(frame)  # Append frame for video

        if isinstance(frame, np.ndarray):
            frames.append(frame)
        else:
            frames.append(np.array(frame))

        if(observation > 130):
           action, _states =  highmodel.predict(observation)
        if(observation < 130 and observation > 70):
           action, _states =  innermodel.predict(observation)
        else:
           action, _states =  lowmodel.predict(observation)
        #Controll Action
        if(action > 0.1):
            action = 0.1
        if(observation < 130):
            action = 0

        observation, reward, terminated, truncated, info = env.step(action)

        #log for dataframe
        log_data.append({
            "action" : action,
            "blood glucose": observation[0],
            "reward": reward,
            "meal": info["meal"],
            "risk": info["risk"],
        })

        logging.info(f'Action taken: {action}, Blood Glucose: {observation[0]}, Reward: {reward}')

        if truncated:
            save_plot_as_video(SAVE_VIDEO, path_to_SimResults_sub_folders, frames, "PlotVideo.mp4")
            save_data_as_csv(SAVE_TO_CSV, path_to_SimResults_sub_folders, log_data, filename="LogData.csv")
            
    env.close()


if __name__ == "__main__":
    main()