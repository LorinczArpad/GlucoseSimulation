# Megkéne nézni a gym es megközelítést

#T1DSimEnv class-ban átírni a reward functiont valami hasonlóra:
#Majd init be is kell ezt azt állítani (itt benne can a reward function is ?)
#spaces a  gym-ből van (ez a tartomány amit az agent próbál kitalálni (ezt hogy kell beállítani az wtf + doksi búvárkodás))
class CustomSimGlucoseEnv(T1DSimEnv):
    def __init__(self, patient, sensor, pump, scenario):
        super().__init__(patient, sensor, pump, scenario)
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([10]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([400]), dtype=np.float32)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        glucose = observation
        if glucose > 180:
            reward = -1
        elif glucose < 70:
            reward = -1
        else:
            reward = 1
        return observation, reward, done, info

#Reward Function(Büntetés ha túl alacsony, vagy túl nagya a vércukorszint az adott stepben ())
#Kicsit hegeszteni kell majd a mostlévő pipeline-on mert az az előző órában mért glucose szint alapján díjjaz
def custom_reward(observation, action):
    glucose = observation['CGM']  
    insulin = action
    if glucose < 70: 
        return -10
    elif glucose > 180: 
        return -10
    else:
        return 1  
# Használat 
# Majd ki kell próbálni
import gym
from stable_baselines3 import PPO
from simglucose.envs.simglucose_gym_env import T1DSimEnv
from stable_baselines3.common.env_util import make_vec_env

#Környezet csak pateinttel?
env = T1DSimEnv(patient_name='adult#001')

# Nem teljesen értem, valami wrapper a env-hez
vec_env = make_vec_env(lambda: env, n_envs=1)

# Model inicializálás (PPO)
model = PPO("MlpPolicy", vec_env, verbose=1)

#Tanítás
model.learn(total_timesteps=50000)

#Mentés
model.save("./Results/test")

# Reseteli a környezetet (pl beállítja az alap szintet (ezt az obs cuccot kapjuk a Controllerben is))
obs = vec_env.reset()

# Elméletileg vizualizálható?
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()

vec_env.close()