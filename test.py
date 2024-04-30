import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import imageio
import numpy as np
from tqdm import tqdm

# Configuration
name_exp = "SAC"
name_model = "models/model_"+name_exp
total_timesteps = 5000000
device = "cuda"
# net_arch=dict(pi=[128, 128, 128],vf=[128, 128, 128])
# policy_kwargs = dict(net_arch=net_arch)

num_steps = 1000
make_gif = True
if make_gif: 
    render_type = "rgb_array" 
else:
    render_type = "human"

env = gym.make("Humanoid-v4", render_mode="rgb_array")

model = SAC.load(name_model, env=env, device=device)#, policy_kwargs=policy_kwargs)

vec_env = model.get_env()
obs = vec_env.reset()

img = model.env.render(mode=render_type)
images = [img]

for i in tqdm(range(num_steps)):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    img = vec_env.render(render_type)
    images.append(img)

if make_gif:
    print("Creating gif")
    fps = 20
    duration = num_steps/fps
    ims = np.array([img for i, img in enumerate(images)])[::3]
    imageio.mimsave("humanoid_rl.gif", ims, duration=duration)