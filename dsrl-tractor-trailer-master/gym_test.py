import random
import time
import os

import PIL.ImageShow
import numpy as np
import gymnasium as gym
from stable_baselines3.sac import SAC
from icecream import ic
# from visualization.visualization import Visualization
from visualization.visualization_with_steering import Visualization

import envs


SEED = 0


env_id = 'TractorTrailer-v0'
env_params = dict(
    tractor_length = 3,
    trailer_length = 10,
    max_linear_acceleration = 1,
    max_steering_angular_velocity = 0.2,
    # parking_success_criterion,
    t_max = 60,
    timestep = 0.01,
    integration_steps = 2,
    # timestep = 0.02,
    dynamics_discretization_method = 'RK4',
    # dynamics_discretization_method = 'Euler1',
)
env = gym.make(
    env_id,
    max_episode_steps=int(env_params['t_max']/env_params['timestep']/env_params['integration_steps']),
    np_random=np.random.default_rng(SEED),
    render_mode='human',
    **env_params,
)
# agent = SAC.load('models/2cxrspej/model.zip', env)

options = {
    'init_states': np.array([20, 10, 0, 1, 0, 100*np.pi/180, 0]) # np.array([25, 13, 0, 1, 0, 0, 0])
}
obs, info = env.reset(options=options)
rollout_return = 0
while True:
    act = np.array([0, 0])
    next_obs, rwd, ter, tru, info = env.step(act)
    rollout_return += rwd
    done = float(ter)
    env.render()
    ic(rwd, ter, tru, info)
    if ter or tru: break
    else: obs = next_obs