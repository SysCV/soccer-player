import torch
import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymapi

from isaacgymenvs.tasks.go1_ball_shoot import Go1BallShoot


class RewardTerms:
    def __init__(self, env: Go1BallShoot):
        self.env = env

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)
    
    def every(self):
        rew_distance = self._reward_ball_goal_dis()
        reward_ball_in_goal = self._reward_success()
        rew_ball_dog_distance = self._reward_ball_dog_dis()
        heading_reward = self._reward_dog_heading_ball()
        ball_z_reward = self._reward_ball_height()
        rew_torque = self._reward_torque()
        rew_have_speed = self._reward_ball_speed()

        # speed_error = torch.sum(torch.square(self.env.ball_lin_vel_xy_world - self.env.commands), dim=1)
        # rew_ball_speed = torch.exp(-speed_error)

        total_reward = heading_reward + rew_distance + rew_have_speed + \
            rew_torque \
        # + ball_z_reward 
        + rew_ball_dog_distance + reward_ball_in_goal

    def _reward_ball_speed(self):
        ball_speed_square = torch.sum(torch.square(self.env.ball_lin_vel_xy_world), dim=1)
        rew_have_speed = torch.tanh(torch.clamp_max(ball_speed_square, self.env.reward_params["ball_speed"]["clip_max"]))
        return rew_have_speed

    def _reward_torque(self):
        rew_torque = torch.sum(torch.square(self.env.torques), dim=1)
        return - rew_torque

    def _reward_ball_height(self):
        ball_z = self.env.ball_pos[:, 2]
        ball_z_reward = torch.exp(ball_z)

    def _reward_dog_heading_ball(self):
        robot_heading = torch.tensor([[1., 0., 0.]] * self.env.base_pos.size(0), device=self.env.root_states.device)
        base_quat_world = quat_rotate(self.env.base_quat, robot_heading)
        base_to_ball_world = self.env.ball_pos - self.env.base_pos
        base_quat_world_xy = base_quat_world[:, :3]
        base_to_ball_world_xy = base_to_ball_world[:, :3]
        base_quat_world_xy = base_quat_world_xy / torch.norm(base_quat_world_xy, dim=1, keepdim=True)
        base_to_ball_world_xy = base_to_ball_world_xy / torch.norm(base_to_ball_world_xy, dim=1, keepdim=True)
        dot_product = torch.sum(base_quat_world_xy * base_to_ball_world_xy, dim=1)
        heading_reward = torch.exp(dot_product) 
        return heading_reward

    def _reward_ball_dog_dis(self):
        ball_dog_distance_error = torch.sum(torch.square(self.env.ball_pos - self.env.base_pos), dim=1)
        ball_dog_distance_error = torch.clamp_min(ball_dog_distance_error, self.env.reward_params["ball_dog_dis"]["clip_min"])
        rew_ball_dog_distance = torch.exp(-ball_dog_distance_error)
        return rew_ball_dog_distance

    def _reward_ball_goal_dis(self):
        ball_goal_distance_error = torch.sum(torch.square(self.env.ball_pos - self.env.goal_pos), dim=1)

        rew_distance = torch.exp(-ball_goal_distance_error)
        return rew_distance

    def _reward_success(self):
        reward_ball_in_goal = torch.zeros_like(self.env.ball_in_goal_now, dtype=torch.float, device=self.env.ball_in_goal_now.device)
        reward_ball_in_goal[self.env.ball_in_goal_now] = 1.
        return reward_ball_in_goal