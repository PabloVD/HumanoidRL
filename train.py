# See more here https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC
import os
import warnings
from stable_baselines3.common.callbacks import CheckpointCallback

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Configuration
name_exp = "SAC"
name_model = "models/model_"+name_exp
total_timesteps = 1000000
device = "cuda"
# net_arch=dict(pi=[128, 128, 128],vf=[128, 128, 128])
# policy_kwargs = dict(net_arch=net_arch)


checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./callbacks/', name_prefix=name_exp)

env = gym.make("Humanoid-v4", render_mode="rgb_array")

if os.path.exists(name_model+".zip"):
    print("Loading previous model")
    #model = PPO.load(name_model, env=env, device=device, policy_kwargs=policy_kwargs)
    model = SAC.load(name_model, env=env, device=device)
else:
    #model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./PPO_logs", device=device, policy_kwargs=policy_kwargs)
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./logs", device=device)
print(model.policy)

model.learn(total_timesteps=total_timesteps, tb_log_name=name_exp, callback=checkpoint_callback )

model.save(name_model)
