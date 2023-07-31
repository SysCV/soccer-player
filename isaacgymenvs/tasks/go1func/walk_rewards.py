import torch
import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymapi

from isaacgymenvs.tasks.go1 import Go1


class RewardTerms:
    def __init__(self, env: Go1):
        self.env = env

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.env.reward_params["tracking_lin_vel"]["sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.env.reward_params["tracking_ang_vel"]["sigma"])

    def _reward_torque(self):
        rew_torque = torch.sum(torch.square(self.env.torques), dim=1)
        return rew_torque
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)
    
    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.last_actions - self.env.actions)
        diff = diff * (self.env.last_actions != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.actions - 2 * self.env.last_actions + self.env.last_last_actions)
        diff = diff * (self.env.last_actions != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    
    def _reward_dof_pos(self):
        # Penalize dof positions
        return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)
    
    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env.dof_vel), dim=1)
    
    def _reward_base_height(self):
        # Penalize base height
        return torch.square(self.env.base_pos[:, 2] - self.env.reward_params["base_height"]["target"])
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        first_contact = (self.env.feet_air_time > 0.) * self.env.contact_state
        rew_airTime = torch.sum((self.env.feet_air_time - self.env.reward_params["feet_air_time"]["baseline"]) * first_contact, dim=1) # reward only on first contact with the ground
        self.env.feet_air_time *= self.env.contact_state
        return rew_airTime
    
    def _reward_lin_vel_z(self):
        # Reward forward velocity
        return torch.square(self.env.base_lin_vel[:, 2])
    

    

