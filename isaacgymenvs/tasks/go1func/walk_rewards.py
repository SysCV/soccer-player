import torch
import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymapi

from isaacgymenvs.tasks.go1 import Go1


# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0.0, -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.0) / 2.0
    return (upper - lower) * r + lower


def get_scale_shift(range):
    scale = 2.0 / (range[1] - range[0])
    shift = (range[1] + range[0]) / 2.0
    return scale, shift


class RewardTerms:
    def __init__(self, env: Go1):
        self.env = env

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1
        )
        # print("command_v", self.env.commands[10:15, :2])
        # print("base_v", self.env.base_lin_vel[10:15, :2])
        return torch.exp(
            -lin_vel_error / self.env.reward_params["tracking_lin_vel"]["sigma"]
        )

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(
            self.env.commands[:, 2] - self.env.base_ang_vel[:, 2]
        )
        return torch.exp(
            -ang_vel_error / self.env.reward_params["tracking_ang_vel"]["sigma"]
        )

    def _reward_torque(self):
        rew_torque = torch.sum(torch.square(self.env.torques), dim=1)
        return rew_torque

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_targets - self.env.actions), dim=1)

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

    def _reward_base_height(self):
        # Penalize base height
        return torch.square(
            self.env.base_pos[:, 2] - self.env.reward_params["base_height"]["target"]
        )

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        first_contact = (self.env.feet_air_time > 0.0) * self.env.contact_state
        rew_airTime = torch.sum(
            (
                self.env.feet_air_time
                - self.env.reward_params["feet_air_time"]["baseline"]
            )
            * first_contact,
            dim=1,
        )  # reward only on first contact with the ground
        rew_airTime *= (
            torch.norm(self.env.commands[:, :2], dim=1) > 0.1
        )  # no reward for zero command
        self.env.feet_air_time *= ~(self.env.contact_state > 0.0)  # here fix a  bug!!
        return rew_airTime

    def _reward_lin_vel_z(self):
        # Reward forward velocity
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

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = (
            self.env.foot_positions - self.env.base_pos.unsqueeze(1)
        )
        # print("foot_pos: ", self.env.foot_positions)
        footsteps_in_body_frame = torch.zeros(
            self.env.num_envs, 4, 3, device=self.env.device
        )
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(
                quat_conjugate(self.env.base_quat), cur_footsteps_translated[:, i, :]
            )

            # nominal positions: [FR, FL, RR, RL]
            # TODO: in body frame is [FL, FR, RL, RR]?
            desired_stance_width = 0.3
            desired_ys_nom = torch.tensor(
                [
                    desired_stance_width / 2,
                    -desired_stance_width / 2,
                    desired_stance_width / 2,
                    -desired_stance_width / 2,
                ],
                device=self.env.device,
            ).unsqueeze(0)

            desired_stance_length = 0.45
            desired_xs_nom = torch.tensor(
                [
                    desired_stance_length / 2,
                    desired_stance_length / 2,
                    -desired_stance_length / 2,
                    -desired_stance_length / 2,
                ],
                device=self.env.device,
            ).unsqueeze(0)

        # print("foot_step: ", footsteps_in_body_frame[:, :, 0:3])

        # raibert offsets
        phases = torch.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.env.frequencies
        x_vel_des = self.env.commands[:, 0:1]
        yaw_vel_des = self.env.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat(
            (desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2
        )

        # print("x_vel: ", x_vel_des)
        # print("phases: ", phases)
        # print("p_desir: ", desired_footsteps_body_frame)

        err_raibert_heuristic = torch.abs(
            desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2]
        )

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
