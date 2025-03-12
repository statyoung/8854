import time
from typing import Dict, Tuple, Any, Union

import pygame
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

# from visualization.visualization import Visualization
from visualization.visualization_with_steering import Visualization
from envs.tractor_trailer_dynamics import TractorTrailer2D


class TractorTrailer(gym.Env):

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30,
    }

    def __init__(
            self,
            tractor_length: float = 3,
            trailer_length: float = 10,
            max_linear_acceleration: float = 1,
            max_steering_angular_velocity: float = 0.2,
            parking_success_criterion: Dict[str, float] = {
                'position': 0.5,
                'heading': 0.05,
                'rel_angle': 0.05,
                'steering': 0.05,
                'velocity': 0.05,
            },
            t_max: float = 30,
            timestep: float = 0.01,
            integration_steps: int = 10,
            dynamics_discretization_method: str = 'RK4',
            reward_params: Dict[str, float] = {
                'weight_p': 0.1,
                'decay_p': 0.1,
                'weight_heading': 0.01,
                'decay_heading': 1,
                'weight_rel_angle': 0.05,
                'decay_rel_angle': 1,
            },
            render_mode: str = None,
            np_random: Union[int, np.random.Generator] = None,
    ):
        
        for key in parking_success_criterion.keys():
            if key not in ['position', 'heading', 'rel_angle', 'steering', 'velocity']:
                raise KeyError(f'Required parking success criterion parameter (:{key}) is missing.')
        for key in reward_params.keys():
            if key not in ['weight_p', 'decay_p', 'weight_heading', 'decay_heading', 'weight_rel_angle', 'decay_rel_angle']:
                raise KeyError(f'Required reward function parameter (:{key}) is missing.')
        assert dynamics_discretization_method in ['RK4', 'Euler1']
        
        self.timestep = timestep
        self.integration_steps = integration_steps
        self.dynamics = TractorTrailer2D(
            tractor_length,
            trailer_length,
            max_linear_acceleration,
            max_steering_angular_velocity,
            parking_success_criterion,
            t_max,
            timestep,
            dynamics_discretization_method,
            reward_params,
        )
        self.observation_space = Box(
            self.dynamics.state_low_,
            self.dynamics.state_high_,
            self.dynamics.state_high_.shape,
            dtype=np.float64,
            seed=np_random,
        )
        self.action_space = Box(
            self.dynamics.action_low_,
            self.dynamics.action_high_,
            self.dynamics.action_high_.shape,
            seed=np_random,
        )
        self.render_mode = render_mode
        if render_mode is not None:
            if render_mode == 'human' or render_mode == 'rgb_array':
                self.renderer = Visualization(
                    'visualization/settings.yaml',
                    tractor_length,
                    trailer_length,
                    x_init=0, y_init=0,
                    yaw_init=(1,0),
                    gamma=0,
                    display=True if render_mode == 'human' else False,
                    fps=self.metadata['render_fps']
                )
            else:
                raise NotImplementedError(f'render_mode (:{render_mode}) is not available.')

    def reset(
            self, 
            seed: int = None,
            options: Dict[str, Any] = {},
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        observation = self.dynamics.reset(options=options)
        info = {}
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        for _ in range(self.integration_steps):
            next_observation, reward_dict, success = self.dynamics.step(action)
            reward = sum([reward_dict[key] for key in reward_dict.keys()])
            terminated = success
            truncated = False
            info = dict()
            info.update(reward_dict)
            if success: break
        return next_observation, reward, terminated, truncated, info
    
    def render(self) -> np.ndarray:  
        x, y = self.dynamics.p_
        yaw_axis = self.dynamics.heading_axis_
        steering_angle = self.dynamics.delta_
        gamma = self.dynamics.lambda_
        image = self.renderer.rend_truck(x, y, yaw_axis, gamma, steering_angle)
        if self.render_mode == 'rgb_array':
            return image

    def close(self):
        pygame.quit()
