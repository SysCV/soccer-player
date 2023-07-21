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
        return - rew_torque

    

