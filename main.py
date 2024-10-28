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

class CustomT1DSimGymnaisumEnv(T1DSimGymnaisumEnv):
    def step(self, action):
        
        observation, reward, terminated, truncated, info = super().step(action)
        blood_glucose = observation[0]  
        target_bg = 120  
        bg_tolerance = 20  

        
        deviation = abs(blood_glucose - target_bg)

        
        if blood_glucose > target_bg + bg_tolerance:
           
            reward -= 5 * (deviation / 10) 
        elif blood_glucose < target_bg - bg_tolerance:
           
            reward -= 5 * (deviation / 10) 
        else:
            
            reward += 5 

            if hasattr(self, 'last_blood_glucose'):
                fluctuation = abs(blood_glucose - self.last_blood_glucose)
                if fluctuation < 10: 
                    reward += 2  
                elif fluctuation >= 10 and fluctuation < 20:  
                    reward -= 1  
            
          
            self.last_blood_glucose = blood_glucose

        return observation, reward, terminated, truncated, info 
    

def main():
    from simglucose.simulation.scenario import CustomScenario
    meal_events = [
        (8, 20), 
        (12, 15),   
        (18, 25),   
    ]
    start_time = datetime(2024, 10, 28, 15, 46, 0)
    meal_scenario = CustomScenario(start_time=start_time, scenario=meal_events)
    register(
        id="simglucose/adolescent2-v0",
        entry_point="main:CustomT1DSimGymnaisumEnv",
        max_episode_steps=1000,
        kwargs={"patient_name": "adolescent#002",
                'custom_scenario': meal_scenario},
    )

    env = gymnasium.make("simglucose/adolescent2-v0", render_mode="human")  # Use "human" mode

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    logging.basicConfig(level=logging.INFO)
    observation, info = env.reset()
    truncated = False
    while(truncated != True):
        env.render()  # Display the environment

        action, _states = model.predict(observation)
        if(action > 0.1):
            action = 0.1
        if(observation < 130):
            action = 0
        observation, reward, terminated, truncated, info = env.step(action)
        logging.info(f'Action taken: {action}, Blood Glucose: {observation[0]}, Reward: {reward}')
        
    
    # Save video with imageio
    
    env.close()


if __name__ == "__main__":
    main()