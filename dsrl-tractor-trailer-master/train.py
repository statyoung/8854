from functools import partial

import numpy as np
import gymnasium as gym
import hydra
from omegaconf import DictConfig
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from icecream import ic

import envs


@hydra.main(version_base=None, config_path='.', config_name='conf.yaml')
def main(cfg: DictConfig):

    run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,
        sync_tensorboard=True,
        monitor_gym=True,
    )

    env_id = 'TractorTrailer-v0'
    env_params = dict(
        tractor_length = cfg.env.tractor_length,
        trailer_length = cfg.env.trailer_length,
        max_linear_acceleration = cfg.env.max_linear_acceleration,
        max_steering_angular_velocity = cfg.env.max_steering_angular_velocity,
        parking_success_criterion = dict(cfg.env.parking_success_criterion),
        t_max = cfg.env.t_max,
        timestep = cfg.env.timestep,
        integration_steps = cfg.env.integration_steps,
        dynamics_discretization_method = cfg.env.dynamics_discretization_method,
        reward_params = dict(cfg.env.reward)
    )
    max_episode_steps = int(env_params['t_max']/env_params['timestep']/env_params['integration_steps'])
    run.config.update(env_params)

    def make_env(seed: int = 0, rank: int = 0):
        def env_fn(seed: int = seed + rank):
            env = gym.make(
                env_id,
                max_episode_steps=max_episode_steps,
                np_random=np.random.default_rng(seed),
                render_mode='rgb_array',
                **env_params,
            )
            env.reset()
            return Monitor(env)
        return env_fn

    n_envs = cfg.train.num_envs
    env = SubprocVecEnv([make_env(seed=i, rank=cfg.train.seed) for i in range(n_envs)])
    env = VecVideoRecorder(
        env,
        f'videos/{run.id}',
        record_video_trigger=lambda x: x % int(max_episode_steps * 10) == 0,
        video_length=max_episode_steps,
    )

    model = SAC(
        'MlpPolicy',
        env,
        policy_kwargs=dict(
            net_arch=dict(cfg.train.net_arch),
        ),
        verbose=1,
        tensorboard_log=f'runs/{run.id}',
        gradient_steps=-1
    )
    model.learn(
        total_timesteps=cfg.train.total_gradient_steps,
        log_interval=4,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_freq=100000,
            model_save_path=f'models/{run.id}',
            verbose=2,
        ),
    )
    model.save(cfg.train.model_save_path)


if __name__ == '__main__':
    main()