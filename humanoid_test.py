import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import imageio
import numpy as np

name_model = "ppo_humanoid"
num_steps = 200
make_gif = True
if make_gif: 
    render_type = "rgb_array" 
else:
    render_type = "human"

env = gym.make("Humanoid-v4", render_mode="rgb_array")

model = PPO.load(name_model, env=env)

vec_env = model.get_env()
obs = vec_env.reset()

img = model.env.render(mode=render_type)
images = [img]

for i in range(num_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    img = vec_env.render(render_type)
    images.append(img)

if make_gif:
    fps = 15
    duration = num_steps/fps
    ims = np.array([img for i, img in enumerate(images)])[::2]
    imageio.mimsave("humanoid_rl.gif", ims, duration=duration)