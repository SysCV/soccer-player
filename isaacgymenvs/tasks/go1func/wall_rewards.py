import torch
import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymapi

from isaacgymenvs.tasks.go1_wall_kicker import Go1WallKicker


# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


class RewardTerms:
    def __init__(self, env: Go1WallKicker):
        self.env = env

    # ------------ guidance reward functions----------------

    def _reward_ball_speed_norm(self):
        ball_speed_square = torch.sum(
            torch.square(self.env.ball_lin_vel_xy_world), dim=1
        )
        rew_have_speed = torch.tanh(
            ball_speed_square / self.env.reward_params["ball_speed_norm"]["sigma"]
        )
        return rew_have_speed

    def _reward_ball_direction_guidance(self):
        # reward for ball speed align with ball-goal vector
        pball_goal = self.env.goal_pos[:, :2] - self.env.ball_pos[:, :2]
        pball_robot = self.env.base_pos[:, :2] - self.env.ball_pos[:, :2]

        angle_diff_goal = torch.atan2(
            self.env.ball_lin_vel_xy_world[:, 1],
            self.env.ball_lin_vel_xy_world[:, 0],
        ) - torch.atan2(pball_goal[:, 1], pball_goal[:, 0])
        angle_diff_in_pi_goal = torch.pow(wrap_to_pi(angle_diff_goal), 2)

        angle_diff_robot = torch.atan2(
            self.env.ball_lin_vel_xy_world[:, 1],
            self.env.ball_lin_vel_xy_world[:, 0],
        ) - torch.atan2(pball_robot[:, 1], pball_robot[:, 0])
        angle_diff_in_pi_robot = torch.pow(wrap_to_pi(angle_diff_robot), 2)
        rew_ball_speed_guidance = torch.exp(
            -1 * torch.minimum(angle_diff_in_pi_robot, angle_diff_in_pi_goal)
        )

        # print("shape of rew_ball_speed_guidance", rew_ball_speed_guidance)
        return rew_ball_speed_guidance

    def _reward_robot_y_guidance(self):
        y_diff = self.env.base_pos[:, 1] - self.env.ball_pos[:, 1]
        rew_robot_y_guidance = torch.exp(
            -1
            * torch.square(y_diff)
            / self.env.reward_params["robot_y_guidance"]["sigma"]
        )
        return rew_robot_y_guidance

    def _reward_dog_heading_ball(self):
        robot_heading = torch.tensor(
            [[1.0, 0.0, 0.0]] * self.env.base_pos.size(0),
            device=self.env.root_states.device,
        )
        base_heading_world = quat_rotate(self.env.base_quat, robot_heading)
        base_to_ball_world = self.env.ball_pos - self.env.base_pos
        base_heading_world_xy = base_heading_world[:, :2]
        base_to_ball_world_xy = base_to_ball_world[:, :2]
        base_heading_world_xy = base_heading_world_xy / torch.norm(
            base_heading_world_xy, dim=1, keepdim=True
        )
        base_to_ball_world_xy = base_to_ball_world_xy / torch.norm(
            base_to_ball_world_xy, dim=1, keepdim=True
        )
        dot_product = torch.sum(base_heading_world_xy * base_to_ball_world_xy, dim=1)
        heading_reward = torch.exp(-1 * (1 - dot_product))
        return heading_reward

    def _reward_ball_dog_dis(self):
        ball_dog_distance_error = torch.sum(
            torch.square(self.env.ball_pos - self.env.base_pos), dim=1
        )
        ball_dog_distance_error = torch.clamp_min(
            ball_dog_distance_error, self.env.reward_params["ball_dog_dis"]["clip_min"]
        )
        rew_ball_dog_distance = torch.exp(
            -1
            / self.env.reward_params["ball_dog_dis"]["sigma"]
            * ball_dog_distance_error
        )
        return rew_ball_dog_distance

    def _reward_ball_goal_dis(self):
        ball_goal_distance_error = torch.sum(
            torch.square(self.env.ball_pos - self.env.goal_pos), dim=1
        )

        rew_distance = torch.exp(
            -1
            / self.env.reward_params["ball_goal_dis"]["sigma"]
            * ball_goal_distance_error
        )
        return rew_distance

    # def _reward_hit_and_switch(self):
    #     reward_index = self.env.ball_in_goal_now & (~self.env.is_back)
    #     self.env.is_back[reward_index] = True
    #     reward_hit = torch.zeros_like(self.env.ball_in_goal_now, dtype=torch.float, device=self.env.ball_in_goal_now.device)
    #     reward_hit[reward_index] = 1.
    #     return reward_hit

    def _reward_hit_wall_and_switch(self):
        reward_index = self.env.ball_near_wall_now & (~self.env.is_back)
        reward_hit = torch.zeros_like(
            self.env.ball_near_wall_now,
            dtype=torch.float,
            device=self.env.ball_near_wall_now.device,
        )
        ball_goal_distance_error = torch.sum(
            torch.square(self.env.ball_pos - self.env.goal_pos), dim=1
        )
        # rew_distance = torch.exp(-ball_goal_distance_error)
        reward_hit[reward_index] = 1.0  # + rew_distance[reward_index]

        bonus_index = (
            ball_goal_distance_error
            < self.env.reward_params["hit_wall_and_switch"]["valid_success_range"] ** 2
        ) & reward_index

        reward_hit[bonus_index] += self.env.reward_params["hit_wall_and_switch"][
            "success_bonus_scale"
        ] / (
            1
            + self.env.reward_params["hit_wall_and_switch"]["hit_times_penalty"]
            * self.env.rebound_times[bonus_index]
        )

        self.env.is_back[reward_index] = True
        self.env.target_reset_buf[:] = False
        self.env.target_reset_buf[bonus_index] = True
        self.env.rebound_times[reward_index] += 1
        self.env.rebound_times[bonus_index] = 0
        return reward_hit

    def _reward_catch_and_switch(self):
        reward_index = self.env.ball_near_robot_now & self.env.is_back
        self.env.is_back[reward_index] = False
        reward_catch = torch.zeros_like(
            self.env.ball_near_robot_now,
            dtype=torch.float,
            device=self.env.ball_near_robot_now.device,
        )
        reward_catch[reward_index] = 1.0
        self.env.rebound_times[reward_index] += 1
        return reward_catch

    def _reward_dog_wall_dis(self):
        dog_wall_dis = torch.abs(self.env.base_pos[:, 0] - self.env.goal_pos[:, 0])
        dog_wall_dis_error = torch.square(
            dog_wall_dis - self.env.reward_params["dog_wall_dis"]["good_dis"]
        )
        rew_dog_wall_dis = torch.exp(-dog_wall_dis_error)
        return rew_dog_wall_dis

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(
            1.0
            * (
                torch.norm(
                    self.env.contact_forces[:, self.env.penalised_contact_indices, :],
                    dim=-1,
                )
                > 0.1
            ),
            dim=1,
        )

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
        diff = torch.square(self.env.last_targets - self.env.targets)
        diff = diff * (self.env.last_targets != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(
            self.env.targets - 2 * self.env.last_targets + self.env.last_last_targets
        )
        diff = diff * (self.env.last_targets != 0)  # ignore first step
        diff = diff * (self.env.last_last_targets != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    def _reward_dof_pos(self):
        # Penalize dof positions
        return torch.sum(
            torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1
        )

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env.dof_vel), dim=1)

    # def _reward_base_height(self):
    #     # Penalize base height
    #     return torch.square(self.env.base_pos[:, 2] - self.env.reward_params["base_height"]["target"])

    def _reward_lin_vel_z(self):
        # Reward z velocity
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(
            self.env.contact_forces[:, self.env.feet_indices, :], dim=-1
        )
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += -(1 - desired_contact[:, i]) * (
                1
                - torch.exp(
                    -1
                    * foot_forces[:, i] ** 2
                    / self.env.reward_params["tracking_contacts_shaped_force"]["sigma"]
                )
            )
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(
            self.env.num_envs, -1
        )
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += -(
                desired_contact[:, i]
                * (
                    1
                    - torch.exp(
                        -1
                        * foot_velocities[:, i] ** 2
                        / self.env.reward_params["tracking_contacts_shaped_vel"][
                            "sigma"
                        ]
                    )
                )
            )
        return reward / 4

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(
            torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt),
            dim=1,
        )
