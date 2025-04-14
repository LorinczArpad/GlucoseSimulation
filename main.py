from pathlib import Path
from PIL import ImageGrab
from colorama import Fore, Style, init as init_colorama
import imageio
import time
import numpy as np
import pandas as pd
import gymnasium
import random
from datetime import timedelta
from gymnasium.envs.registration import register
from simglucose.simulation.scenario import CustomScenario
from datetime import datetime
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from simglucose.envs import T1DSimGymnaisumEnv
import logging
from simglucose.simulation.scenario import CustomScenario

def create_and_get_directory_for_sim_results(kwargs):
    #Create folder with patient name
    
    patient_name = kwargs.get("patient_name")
    base_folder = Path("SimResults/" + str(patient_name) + "_00")
    modelName = "A2C"
    counter = 0
    while base_folder.exists():
        suffix = f"_{counter:02d}"
        base_folder = Path(f"SimResults/{modelName}_{patient_name}{suffix}")
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

    patient_name = "adult#002"
      
    #breakfast = (random.randint(6, 9), random.randint(40, 80))  
    #snack1 = (random.randint(10, 11), random.randint(10, 25))  
    #lunch = (random.randint(12, 15), random.randint(50, 100))    
    #snack2 = (random.randint(16, 17), random.randint(10, 25))    
    #dinner = (random.randint(18, 21), random.randint(50, 90))

    def generate_meals(total_carb):
        # Random arányok generálása (összegük = 1)
        ratios = [random.uniform(0.2, 0.4), random.uniform(0.3, 0.5), random.uniform(0.2, 0.4)]
        total = sum(ratios)
        normalized = [r / total for r in ratios]

        # Carb gramm kiszámítása és kerekítése
        breakfast_carb = round(normalized[0] * total_carb)
        lunch_carb = round(normalized[1] * total_carb)
        dinner_carb = total_carb - breakfast_carb - lunch_carb 

        breakfast = (random.randint(6, 9), breakfast_carb)
        lunch = (random.randint(12, 15), lunch_carb)
        dinner = (random.randint(18, 21), dinner_carb)


        return breakfast, lunch, dinner

    
    breakfast, lunch, dinner = generate_meals(random.randint(100, 140))
    snack1=(random.randint(10,11),random.randint(8,11))
    snack2=(random.randint(16,17),random.randint(8,11)) 

    
    meal_events = [
        (breakfast), 
        (snack1),   
        (dinner),
        (snack2),
        (lunch)   
    ]

    start_time = datetime(2025, 1, 1, 0, 0, 0)

    meal_scenario = CustomScenario(start_time=start_time, scenario=meal_events)

    base_kwargs = {"patient_name": patient_name, 'custom_scenario': meal_scenario}

    path_to_SimResults_sub_folders = create_and_get_directory_for_sim_results(base_kwargs)
    print(f"Folder created: {path_to_SimResults_sub_folders}")
    
    register(
        id="simglucose/adolescent2-v0",
        entry_point="customEnviroments:CustomT1DSimGymnaisumEnv",
        max_episode_steps=480,
        kwargs=base_kwargs,
    )
    register(
        id="simglucose/adolescent2-v0-low",
        entry_point="customEnviroments:LowGlucoseEnv",
        max_episode_steps=480,
        kwargs=base_kwargs,
    )
    register(
        id="simglucose/adolescent2-v0-high",
        entry_point="customEnviroments:HighGlucoseEnv",
        max_episode_steps=480,
        kwargs=base_kwargs,
    )
    register(
        id="simglucose/adolescent2-v0-inner",
        entry_point="customEnviroments:InnerGlucoseEnv",
        max_episode_steps=480,
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
    bw = env.patient._params["BW"].iloc[0]
    #Models
    lowmodel = A2C("MlpPolicy",lowenv,learning_rate=0.001,verbose=1)
    innermodel = A2C("MlpPolicy", innerenv,learning_rate=0.001,  verbose=1)
    highmodel = A2C("MlpPolicy", highenv,learning_rate=0.001,  verbose=1)
    #Training
    TimeSteps = 10000
    lowmodel.learn(total_timesteps=TimeSteps)
    innermodel.learn(total_timesteps=TimeSteps)
    highmodel.learn(total_timesteps=TimeSteps)
    #model = PPO("MlpPolicy", env, verbose=1)
    #model.learn(total_timesteps=500)

    logging.basicConfig(level=logging.INFO)

    frames = []
    log_data = []

    observation, info = env.reset()
    truncated = False
    risk = 0
    current_time = start_time
    print("CHO intakes:")
    print(Fore.BLUE+f"   Breakfast: at {breakfast[0]} o'clock {breakfast[1]}g")
    print(Fore.BLUE+f"   Lunch: at {lunch[0]} o'clock {lunch[1]}g")
    print(Fore.BLUE+f"   Dinner: at {dinner[0]} o'clock {dinner[1]}g")
    print(Fore.RESET)

    while(current_time < start_time + timedelta(hours=24) and not truncated):
        env.render()
        current_time += timedelta(minutes=3)
        # Capture screen
        screen = ImageGrab.grab()
        frame = np.array(screen)
        frames.append(frame)  # Append frame for video

        if(observation > 130):
           action, _states =  highmodel.predict(observation)
        if(observation < 130 and observation > 70):
           action, _states =  innermodel.predict(observation)
        else:
           action, _states =  lowmodel.predict(observation)
        
        #Controll Action
        
       
        coefficient=1
        if(risk > 1):
            coefficient=risk*0.45
        
        if(action > 0.1):
              action = 0.1*coefficient
        if(observation < 125):
             action = 0
       
        action = action*coefficient
        if(action > 4):
            action = 3.5
     
        observation, reward, terminated, truncated, info = env.step(action)
        risk= info["risk"]

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