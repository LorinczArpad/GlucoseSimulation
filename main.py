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

class CustomT1DSimGymnaisumEnv(T1DSimGymnaisumEnv):
    def step(self, action):
        # Call the original step method to get the observation, reward, etc.
        observation, reward, terminated, truncated, info = super().step(action)
        blood_glucose = observation[0]  # Assuming the first observation is the blood glucose level
        target_bg = 120  # Ideal blood glucose level
        bg_tolerance = 20  # Acceptable range around the target

        # Calculate deviation from target
        deviation = abs(blood_glucose - target_bg)

        # Penalize based on how far the blood glucose is from the target
        if blood_glucose > target_bg + bg_tolerance:
            # Too high
            reward -= 5 * (deviation / 10)  # Penalize more for higher deviations
        elif blood_glucose < target_bg - bg_tolerance:
            # Too low
            reward -= 5 * (deviation / 10)  # Penalize more for lower deviations
        else:
            # Within acceptable range
            reward += 5  # Base reward for being in the desired range

            # Add bonus for staying stable in the desired range
            # Assuming `last_blood_glucose` is a previously saved observation
            if hasattr(self, 'last_blood_glucose'):
                fluctuation = abs(blood_glucose - self.last_blood_glucose)
                if fluctuation < 10:  # Small fluctuation
                    reward += 2  # Reward for stability
                elif fluctuation >= 10 and fluctuation < 20:  # Moderate fluctuation
                    reward -= 1  # Small penalty for larger fluctuation
            
            # Update last blood glucose for the next step
            self.last_blood_glucose = blood_glucose

        return observation, reward, terminated, truncated, info  # Return all values
    
# def main():
#     register(
#         id="simglucose/adolescent2-v0",
#         entry_point="main:CustomT1DSimGymnaisumEnv",
#         max_episode_steps=100,
#         kwargs={"patient_name": "adolescent#002"},
#         )

#     env = gymnasium.make("simglucose/adolescent2-v0", render_mode="human")

#     model = PPO("MlpPolicy", env, verbose=1)
#     model.learn(total_timesteps=10000)
    
#     observation, info = env.reset()
#     for t in range(200):
#         env.render()
#         print(env.action_space)
#         action, _states = model.predict(observation)
#         print(action)
#         observation, reward, terminated, truncated, info = env.step(action)
#         print(observation)
#         # print(
#         #     f"Step {t}: observation {observation}, reward {reward}, terminated {terminated}, truncated {truncated}, info {info}"
#         # )

        
#         if terminated or truncated:
#             print("Episode finished after {} timesteps".format(t + 1))
#             break

def main():
    register(
        id="simglucose/adolescent2-v0",
        entry_point="main:CustomT1DSimGymnaisumEnv",
        max_episode_steps=100,
        kwargs={"patient_name": "adolescent#002"},
    )

    env = gymnasium.make("simglucose/adolescent2-v0", render_mode="human")  # Use "human" mode

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    
    observation, info = env.reset()
    frames = []

    for t in range(200):
        env.render()  # Display the environment
        time.sleep(0.1)  # Small delay to allow rendering

        # Capture screen
        screen = ImageGrab.grab()
        frame = np.array(screen)
        frames.append(frame)  # Append frame for video

        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print("Episode finished after {} timesteps".format(t + 1))
            break
    
    # Save video with imageio
    for i in range(100):
        file_name = Path("SimResults\env_simulation_" + str(i) + ".mp4")
        if not file_name.is_file():
            break

    if file_name.is_file():
        file_name = "env_simulation_OVERFLOW.mp4"
        
    if frames.count > 1:
        imageio.mimsave(file_name, frames, format="mp4", fps=5)  # Save frames as video

    env.close()
    print("Video saved as " + str(file_name))

if __name__ == "__main__":
    main()