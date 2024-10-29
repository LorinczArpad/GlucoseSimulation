from pathlib import Path
from PIL import ImageGrab
import imageio
import time
import numpy as np
import gymnasium
from gymnasium.envs.registration import register
from simglucose.simulation.scenario import CustomScenario
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from simglucose.envs import T1DSimGymnaisumEnv
import logging
from simglucose.simulation.scenario import CustomScenario

def main():
    
    meal_events = [
        (8, 20), 
        (12, 15),   
        (18, 25),   
    ]
    start_time = datetime(2024, 10, 28, 15, 46, 0)
    meal_scenario = CustomScenario(start_time=start_time, scenario=meal_events)
    register(
        id="simglucose/adolescent2-v0",
        entry_point="customEnviroments:CustomT1DSimGymnaisumEnv",
        max_episode_steps=1000,
        kwargs={"patient_name": "adolescent#002",
                'custom_scenario': meal_scenario},
    )
    register(
        id="simglucose/adolescent2-v0-low",
        entry_point="customEnviroments:LowGlucoseEnv",
        max_episode_steps=1000,
        kwargs={"patient_name": "adolescent#002",
                'custom_scenario': meal_scenario},
    )
    register(
        id="simglucose/adolescent2-v0-high",
        entry_point="customEnviroments:HighGlucoseEnv",
        max_episode_steps=1000,
        kwargs={"patient_name": "adolescent#002",
                'custom_scenario': meal_scenario},
    )
    register(
        id="simglucose/adolescent2-v0-inner",
        entry_point="customEnviroments:InnerGlucoseEnv",
        max_episode_steps=1000,
        kwargs={"patient_name": "adolescent#002",
                'custom_scenario': meal_scenario},
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

    observation, info = env.reset()
    truncated = False
    while(truncated != True):
        env.render()  

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
        logging.info(f'Action taken: {action}, Blood Glucose: {observation[0]}, Reward: {reward}')
    env.close()


if __name__ == "__main__":
    main()