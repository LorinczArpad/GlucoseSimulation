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
import ModelAndEnviromentHelper

import pkg_resources



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

    
    '''breakfast, lunch, dinner = generate_meals(random.randint(100, 140))
    snack1=(random.randint(10,11),random.randint(8,11))
    snack2=(random.randint(16,17),random.randint(8,11)) 

    
    meal_events = [
        (breakfast), 
        (snack1),   
        (dinner),
        (snack2),
        (lunch)   
    ]
    '''
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    
        
    patient_params_file = pkg_resources.resource_filename("simglucose", "params/vpatient_params.csv")

    # Load the parameters
    patient_params = pd.read_csv(patient_params_file)

    # Get body weight for adult#002
    patient_name = "adult#002"
    bw = patient_params[patient_params["Name"] == patient_name]["BW"].iloc[0]
    
    meal_events = ModelAndEnviromentHelper.generaltnap(bw)
    
    meals = [(event[2], event[0]) for event in meal_events]
    meal_scenario = CustomScenario(start_time=start_time, scenario=meals)

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
    

    print(f"Body weight for {patient_name}: {bw} kg")
    
    tuner = ModelAndEnviromentHelper.HyperparameterTuner(
    low_env=lowenv,
    inner_env=innerenv,
    high_env=highenv,
    n_trials=50,
    timesteps=500,
    n_eval_episodes=5
    )

    # Run tuning
    best_params = tuner.tune_all()

    # Save results
    tuner.save_results("a2c_tuning_results.txt")
    #Models
    '''
    lowmodel = A2C("MlpPolicy",lowenv,learning_rate=0.001,verbose=1)
    innermodel = A2C("MlpPolicy", innerenv,learning_rate=0.001,  verbose=1)
    highmodel = A2C("MlpPolicy", highenv,learning_rate=0.001,  verbose=1)
    '''
    lowmodel = A2C(
        policy="MlpPolicy",
        env=lowenv,
        learning_rate=best_params["lowmodel"]["params"]["learning_rate"],
        n_steps=best_params["lowmodel"]["params"]["n_steps"],
        vf_coef=best_params["lowmodel"]["params"]["vf_coef"],
        ent_coef=best_params["lowmodel"]["params"]["ent_coef"],
        max_grad_norm=best_params["lowmodel"]["params"]["max_grad_norm"],
        gae_lambda=best_params["lowmodel"]["params"]["gae_lambda"],
        gamma=best_params["lowmodel"]["params"]["gamma"],
        policy_kwargs={"net_arch": best_params["lowmodel"]["net_arch"]},
        verbose=1
    )
    innermodel = A2C(
        policy="MlpPolicy",
        env=innerenv,
        learning_rate=best_params["innermodel"]["params"]["learning_rate"],
        n_steps=best_params["innermodel"]["params"]["n_steps"],
        vf_coef=best_params["innermodel"]["params"]["vf_coef"],
        ent_coef=best_params["innermodel"]["params"]["ent_coef"],
        max_grad_norm=best_params["innermodel"]["params"]["max_grad_norm"],
        gae_lambda=best_params["innermodel"]["params"]["gae_lambda"],
        gamma=best_params["innermodel"]["params"]["gamma"],
        policy_kwargs={"net_arch": best_params["innermodel"]["net_arch"]},
        verbose=1
    )
    highmodel = A2C(
        policy="MlpPolicy",
        env=highenv,
        learning_rate=best_params["highmodel"]["params"]["learning_rate"],
        n_steps=best_params["highmodel"]["params"]["n_steps"],
        vf_coef=best_params["highmodel"]["params"]["vf_coef"],
        ent_coef=best_params["highmodel"]["params"]["ent_coef"],
        max_grad_norm=best_params["highmodel"]["params"]["max_grad_norm"],
        gae_lambda=best_params["highmodel"]["params"]["gae_lambda"],
        gamma=best_params["highmodel"]["params"]["gamma"],
        policy_kwargs={"net_arch": best_params["highmodel"]["net_arch"]},
        verbose=1
    )
    #Training
    TimeSteps = 500
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
    for meal in meals:
        print(Fore.BLUE+f"   Meal : at {meal[0]} o'clock {meal[1]}g")
        print(Fore.RESET)

    
    insulin_timestamps = []
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
            coefficient=risk*1.5
        
        if(action > 0.1):
              action = 0.1*coefficient
        if(observation < 125):
             action = 0
       
        action = action*coefficient
        if(action > 4):
            action = 3.5
   
        
        two_hour_ago = current_time - timedelta(hours=2)
        insulin_timestamps = [t for t in insulin_timestamps if t > two_hour_ago]

        #ha már 3 vagy több inzulin beadás történt az elmúlt órában akkor tiltjuk a következőt
        if len(insulin_timestamps) >= 3:
            action = 0
            print(Fore.RED + f"Dosing prohibited! ({len(insulin_timestamps)} / 3)")
            print(Fore.RESET)
        else:
            if action > 0:
                insulin_timestamps.append(current_time)
                print(Fore.YELLOW + f"Inzulin was injected: {current_time.strftime('%H:%M')}")
                print(Fore.CYAN + f"Insulin injections in the last 2 hour: {len(insulin_timestamps)} / 3")
                print(Fore.RESET)

     
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