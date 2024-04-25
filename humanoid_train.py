# See more here https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

name_exp = "PPO"
name_model = "ppo_humanoid"
total_timesteps = 1000000
device = "cpu"

env = gym.make("Humanoid-v4", render_mode="rgb_array")

if os.path.exists(name_model+".zip"):
    print("Loading previous model")
    model = PPO.load(name_model, env=env, device=device)
else:
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./PPO_logs", device=device)
#print(model.policy)

model.learn(total_timesteps=total_timesteps, tb_log_name=name_exp )

model.save(name_model)
