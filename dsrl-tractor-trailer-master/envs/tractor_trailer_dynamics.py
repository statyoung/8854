from typing import Tuple, Dict

import numpy as np
from numpy import sin, cos, tan, sqrt, pi, clip
from numpy.random import rand
from numpy.linalg import norm

from icecream import ic


DEG2RAD = pi/180

class TractorTrailer2D:

    def __init__(
            self,
            tractor_length: float,
            trailer_length: float,
            max_linear_acceleration: float,
            max_steering_angular_velocity: float,
            parking_success_criterion: Dict[str, float],
            t_max: float,  # time horizon of the environment
            timestep: float,  # time interval between consequent integrations
            dynamics_discretization_method: str,  # ['RK4', 'Euler1']
            reward_params: Dict[str, float],
    ):
        ############################# simulation parameters
        self.D_ = tractor_length
        self.L_ = trailer_length
        self.T_max_ = t_max
        self.dt_ = timestep
        self.parking_success_criterion = parking_success_criterion
        ############################# state variables
        self.p_ = None              ## 2D 
        self.heading_axis_ = None   ## 2D unit vector to avoid singularity when single scalar 'theta' is used. 
        self.v_ = None              ## 1D
        self.delta_ = None          ## 1D
        self.lambda_ = None         ## 1D
        
        goal_x = 25
        goal_y = self.L_ + 3
        self.goal_p_ = np.array([goal_x, goal_y])

        ############################## reward parameters
        self.weight_p = reward_params['weight_p']
        self.decay_p = reward_params['decay_p']
        self.weight_heading = reward_params['weight_heading']
        self.decay_heading = reward_params['decay_heading']
        self.weight_rel_angle = reward_params['weight_rel_angle']
        self.decay_rel_angle = reward_params['decay_rel_angle']
        
        ############################# action_space
        self.action_high_ = np.array([max_linear_acceleration, max_steering_angular_velocity])    ##### maximal (linear acceleration along heading angle, steering angular velocity) pair. this can be tuned
        self.action_low_ = -self.action_high_
        
        ############################# admissible state space
        rel_angle_lim = 50
        self.state_high_ = np.array(
                                        [
                                            50.0, 50.0,             #### (x, y) pos
                                            1.0, 1.0,               #### (b1_x, b1_y) heading axis 
                                            np.inf,                 #### v_max for this problem
                                            rel_angle_lim*DEG2RAD,           #### tire steering angle MUST BE SMALLER THAN 'PI/2' 
                                            rel_angle_lim*DEG2RAD            #### relative heading angle
                                        ]
                                    )
        
        self.state_low_ = np.array(
                                        [
                                            0.0, 0.0,               #### (x, y) pos
                                            -1.0, -1.0,             #### (b1_x, b1_y) heading axis 
                                            -np.inf,                #### v_max for this problem
                                            -rel_angle_lim*DEG2RAD,          #### tire steering angle MUST BE SMALLER THAN 'PI/2' 
                                            -rel_angle_lim*DEG2RAD           #### relative heading angle
                                        ]
                                    )
        
        self.DynamicsDiscretizationMethod = dynamics_discretization_method
        ####### available options for dynamics 
        if (dynamics_discretization_method != 'RK4'):
            if (dynamics_discretization_method == 'Euler1'):
                self.DynamicsDiscretizationMethod = dynamics_discretization_method
            else:
                self.DynamicsDiscretizationMethod = 'RK4'
        
    def step(self, u: np.array) -> Tuple[np.ndarray, Dict[str, float], bool]:
        
        if (self.DynamicsDiscretizationMethod == 'Euler1'): ## Euler 1st order 
            cur_state = np.hstack((self.p_, self.heading_axis_, self.v_, self.delta_, self.lambda_))
            self.p_, self.heading_axis_, self.v_, self.delta_, self.lambda_ = self.DynamicsOneStepHelper(cur_state, self.dt_, u)
        
        else: ## Runge-Kutta 4th order method
            cur_state = np.hstack((self.p_, self.heading_axis_, self.v_, self.delta_, self.lambda_))
            temp_p_1_, temp_heading_axis_1_, temp_v_1_, temp_delta_1_, temp_lambda_1_ = self.DynamicsOneStepHelper(cur_state, 0.5*self.dt_, u)
            cur_state = np.hstack((temp_p_1_, temp_heading_axis_1_, temp_v_1_, temp_delta_1_, temp_lambda_1_))
            temp_p_2_, temp_heading_axis_2_, temp_v_2_, temp_delta_2_, temp_lambda_2_ = self.DynamicsOneStepHelper(cur_state, 0.5*self.dt_, u)
            cur_state = np.hstack((temp_p_2_, temp_heading_axis_2_, temp_v_2_, temp_delta_2_, temp_lambda_2_))
            temp_p_3_, temp_heading_axis_3_, temp_v_3_, temp_delta_3_, temp_lambda_3_ = self.DynamicsOneStepHelper(cur_state, self.dt_, u)
            
            rhs_lambda_1_ = -self.v_/self.L_*sin(self.lambda_) - self.v_/self.D_*tan(self.delta_)
            rhs_p_1_ = self.v_*self.heading_axis_
            rhs_heading_axis_1_ = self.v_/self.D_*tan(self.delta_)
            
            rhs_lambda_2_ = -temp_v_1_/self.L_*sin(temp_lambda_1_) - temp_v_1_/self.D_*tan(temp_delta_1_)
            rhs_p_2_ = temp_v_1_*temp_heading_axis_1_
            rhs_heading_axis_2_ = temp_v_1_/self.D_*tan(temp_delta_1_)
            
            rhs_lambda_3_ = -temp_v_2_/self.L_*sin(temp_lambda_2_) - temp_v_2_/self.D_*tan(temp_delta_2_)
            rhs_p_3_ = temp_v_2_*temp_heading_axis_2_
            rhs_heading_axis_3_ = temp_v_2_/self.D_*tan(temp_delta_2_)
            
            rhs_lambda_4_ = -temp_v_3_/self.L_*sin(temp_lambda_3_) - temp_v_3_/self.D_*tan(temp_delta_3_)
            rhs_p_4_ = temp_v_3_*temp_heading_axis_3_
            rhs_heading_axis_4_ = temp_v_3_/self.D_*tan(temp_delta_3_)
            
            self.lambda_ = clip(self.lambda_ + self.dt_/6*(rhs_lambda_1_ + 2*rhs_lambda_2_ + 2*rhs_lambda_3_ +rhs_lambda_4_), self.state_low_[-1], self.state_high_[-1]  )
            self.p_ = clip(self.p_ + self.dt_/6*(rhs_p_1_ + 2*rhs_p_2_ + 2*rhs_p_3_ + rhs_p_4_), self.state_low_[:2], self.state_high_[:2]) 
            self.heading_axis_ = rotation_helper(self.heading_axis_, self.dt_/6*(rhs_heading_axis_1_ + 2*rhs_heading_axis_2_ + 2*rhs_heading_axis_3_ + rhs_heading_axis_4_))
            self.v_ = clip(self.v_ + self.dt_*u[0], self.state_low_[4], self.state_high_[4])              ## 1D
            self.delta_ = clip(self.delta_ + self.dt_*u[1], self.state_low_[5], self.state_high_[5] )         ## 1D
        
        reward_dict, success = self.compute_reward()
        
        return np.hstack((self.p_, self.heading_axis_, self.v_, self.delta_, self.lambda_)), reward_dict, success
    
    def compute_reward(self) -> float:
        success = self.IsParkingSucceded()
        reward_dict = dict(
            r_success = 100 if success else 0,
            r_p = self.weight_p * np.exp(- self.decay_p * norm(self.goal_p_ - self.p_)),
            r_heading = self.weight_heading * np.exp(- self.decay_heading * (1-self.heading_axis_@np.array([0,1]))),
            r_rel_angle = self.weight_rel_angle * np.exp(- self.decay_rel_angle * self.lambda_)
        )
        return reward_dict, success
        
    def IsParkingSucceded(self) -> bool:
        return_bool = (
                        ( norm(self.goal_p_ - self.p_) < self.parking_success_criterion['position'])             ######## when truck is close enough the goal position
                        and ( 1-self.heading_axis_@np.array([0,1]) < self.parking_success_criterion['heading'])      ######## when the heading axis is aligned to the upper direction
                        and ( norm(self.lambda_) < self.parking_success_criterion['rel_angle'] )                       ######## when the relative angle is close enough to 0
                        # and ( norm(self.delta_) < self.parking_success_criterion['steering'] )                        ######## when the tire steering angle is close to the upper direction
                        # and ( norm(self.v_) < self.parking_success_criterion['velocity'] )                               ######## when the truck is slow enough
                        )
        return return_bool
    
    def DynamicsOneStepHelper(self, state, dt, u):
        temp_lambda_ = clip(self.lambda_ + dt*(-state[4]/self.L_*sin(state[-1]) - state[4]/self.D_*tan(state[5])), self.state_low_[-1], self.state_high_[-1]  )
        temp_p_ = clip(self.p_ + dt*state[4]*state[2:4], self.state_low_[:2], self.state_high_[:2]) 
        temp_heading_axis_ = rotation_helper(self.heading_axis_, dt*state[4]/self.D_*tan(state[5]))
        temp_v_ = clip(self.v_ + dt*u[0], self.state_low_[4], self.state_high_[4])              ## 1D
        temp_delta_ = clip(self.delta_ + dt*u[1], self.state_low_[5], self.state_high_[5] )         ## 1D
        return temp_p_, temp_heading_axis_, temp_v_, temp_delta_, temp_lambda_
    
    def reset(self, options: dict = {}) -> np.ndarray:
            
        ############################# state variables are initialized near origin
        # self.p_ = 3*rand(2) + np.array([3, self.D_+self.L_ + 1])                                      ## [3, 6] 
        self.p_ = 10*rand(2) + np.array([20, self.D_+self.L_ + 10])                                      ## [20, 20] ~ [30, 30] 
        
        # theta_ = pi*(2*rand()-1)                                        ## [-pi, pi]
        theta_ = 20*DEG2RAD*(2*rand()-1)                                  ## [-20DEG, 20DEG]
        self.heading_axis_ = np.array([-sin(theta_), cos(theta_)])    ## 2D unit vector to avoid singularity when single scalar 'theta' is used. 
        self.v_ = 0.5*(2*rand()-1)                                      ## [-0.5, 0.5]
        self.delta_ = 10*DEG2RAD*(2*rand()-1)                                ## [-10deg, 10deg]
        self.lambda_ = 10*DEG2RAD*(2*rand()-1)     

        if options.get('init_states', None) is not None:
            states = options['init_states']
            self.p_ = states[0:2]
            self.heading_axis_ = states[2:4]
            self.v_ = states[4]
            self.delta_ = states[-2]
            self.lambda_ = states[-1]
        
        return np.hstack((self.p_, self.heading_axis_, self.v_, self.delta_, self.lambda_))
    
def rotation_helper(axis, angle):
    return np.array([ [ cos(angle), -sin(angle)], [sin(angle), cos(angle)] ])@axis