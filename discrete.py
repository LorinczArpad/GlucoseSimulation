import gym
from gym import spaces
from gym.envs.registration import register
from stable_baselines3 import DQN

def main():
    # Register the SimGlucose environment
    register(
        id='simglucose-adolescent2-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#002'}
    )

    # Create the base environment
    env = gym.make('simglucose-adolescent2-v0')

    # Define the custom action wrapper
    class DiscreteActionWrapper(gym.ActionWrapper):
        def __init__(self, env):
            super(DiscreteActionWrapper, self).__init__(env)
            self.action_space = spaces.Discrete(3)  # 3 discrete actions
            self.action_mapping = {0: 0.0, 1: 0.1, 2: 0.4}  # Map to continuous values

        def action(self, action):
            return [self.action_mapping[action]]  # Return action as a list

    # Apply the wrapper to the environment
    env = DiscreteActionWrapper(env)

    # Initialize and train the DQN model
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Reset the environment for simulation
    observation = env.reset()
    done = False

    # Simulation loop
    while not done:
        # Render the environment (if supported)
        env.render()

        # Predict action using the trained model
        action, _states = model.predict(observation)

        # Step through the environment
        observation, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")

    # Close the environment after simulation
    env.close()

if __name__ == "__main__":
    main()