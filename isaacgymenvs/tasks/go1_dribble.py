# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# ======== Asset info unitree go1: ========
# Got 17 bodies, 16 joints, and 12 DOFs

# Bodies:
# ['base', 'trunk', 'FL_hip', 'FL_thigh_shoulder', 'FL_thigh', 'FL_calf', 'FL_foot', 'FR_hip', 'FR_thigh_shoulder', 'FR_thigh', 'FR_calf', 'FR_foot', 'RL_hip', 'RL_thigh_shoulder', 'RL_thigh', 'RL_calf', 'RL_foot', 'RR_hip', 'RR_thigh_shoulder', 'RR_thigh', 'RR_calf', 'RR_foot', 'imu_link']

# Bodies:
#   0: 'base'
#   1: 'FL_hip'
#   2: 'FL_thigh'
#   3: 'FL_calf'
#   4: 'FL_foot'
#   5: 'FR_hip'
#   6: 'FR_thigh'
#   7: 'FR_calf'
#   8: 'FR_foot'
#   9: 'RL_hip'
#  10: 'RL_thigh'
#  11: 'RL_calf'
#  12: 'RL_foot'
#  13: 'RR_hip'
#  14: 'RR_thigh'
#  15: 'RR_calf'
#  16: 'RR_foot'

# Joints:
#   0: 'FL_hip_joint' (Revolute)
#   1: 'FL_thigh_joint' (Revolute)
#   2: 'FL_calf_joint' (Revolute)
#   3: 'FL_foot_fixed' (Fixed)
#   4: 'FR_hip_joint' (Revolute)
#   5: 'FR_thigh_joint' (Revolute)
#   6: 'FR_calf_joint' (Revolute)
#   7: 'FR_foot_fixed' (Fixed)
#   8: 'RL_hip_joint' (Revolute)
#   9: 'RL_thigh_joint' (Revolute)
#  10: 'RL_calf_joint' (Revolute)
#  11: 'RL_foot_fixed' (Fixed)
#  12: 'RR_hip_joint' (Revolute)
#  13: 'RR_thigh_joint' (Revolute)
#  14: 'RR_calf_joint' (Revolute)
#  15: 'RR_foot_fixed' (Fixed)

# DOFs:
#   0: 'FL_hip_joint' (Rotation)
#   1: 'FL_thigh_joint' (Rotation)
#   2: 'FL_calf_joint' (Rotation)
#   3: 'FR_hip_joint' (Rotation)
#   4: 'FR_thigh_joint' (Rotation)
#   5: 'FR_calf_joint' (Rotation)
#   6: 'RL_hip_joint' (Rotation)
#   7: 'RL_thigh_joint' (Rotation)
#   8: 'RL_calf_joint' (Rotation)
#   9: 'RR_hip_joint' (Rotation)
#  10: 'RR_thigh_joint' (Rotation)
#  11: 'RR_calf_joint' (Rotation)

import numpy as np
import os
import torch
import math
import matplotlib.pyplot as plt

from gym import spaces

import wandb

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgym import gymutil

from isaacgymenvs.tasks.base.vec_task import VecTask

from isaacgymenvs.tasks.curricula.curriculum_torch import RewardThresholdCurriculum

from isaacgymenvs.utils.torch_jit_utils import calc_heading


def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


from typing import Dict, Any, Tuple, List, Set


class Go1Dribbler(VecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.totall_episode = 0
        self.success_episode = 0

        self.command_squence = torch.zeros(50 * 14)
        self.velocity_squence = torch.zeros(50 * 14)
        self.plot_step = 0

        self.cfg = cfg

        self.wandb_extra_log = self.cfg["env"]["wandb_extra_log"]
        self.wandb_log_period = self.cfg["env"]["wandb_extra_log_period"]

        # cam pic
        self.have_cam_window = self.cfg["env"]["cameraSensorPlt"]
        self.pixel_obs = self.cfg["env"]["pixel_observations"]["enable"]
        # print("pixel_obs:", self.pixel_obs)
        if self.have_cam_window:
            _, self.ax = plt.subplots()
            plt.axis("off")
        self.add_real_ball = self.cfg["task"]["target_ball"]

        self.do_rand = self.cfg["env"]["randomization"]
        self.random_frec = self.cfg["env"]["randomization_freq"]

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.hip_addtional_scale = self.cfg["env"]["control"]["hipAddtionalScale"]

        # reward scales
        self.postive_reward_ji22 = self.cfg["env"]["rewards"][
            "only_positive_rewards_ji22_style"
        ]
        self.sigma_rew_neg = self.cfg["env"]["rewards"]["sigma_rew_neg"]
        self.reward_scales = self.cfg["env"]["rewards"]["rewardScales"]
        self.reward_params = self.cfg["env"]["rewards"]["rewardParams"]

        # termination conditions
        self.robot_ball_max = self.cfg["env"]["terminateCondition"]["robot_ball_max"]
        self.ball_speed_max = self.cfg["env"]["terminateCondition"]["ball_speed_max"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]

        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # ball params
        self.ball_init_pos = self.cfg["env"]["ballInitState"]["pos"]
        self.ball_mass = self.cfg["env"]["ballInitState"]["mass"]
        self.ball_rand_pos_range = self.cfg["env"]["ballInitState"]["randomPosRange"]

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # state observation
        numObservations = 0
        print(self.cfg["env"]["state_observations"])
        for v in self.cfg["env"]["state_observations"].values():
            numObservations += v
        self.cfg["env"]["numObservations"] = numObservations

        self.vision_receive_prob = self.cfg["env"]["vision_receive_prob"]

        # noise
        self.add_noise = self.cfg["env"]["add_noise"]
        noise_list = [
            torch.ones(
                self.cfg["env"]["state_observations"]["projected_gravity"],
                device=rl_device,
            )
            * self.cfg["env"]["obs_noise"]["projected_gravity"],
            torch.ones(
                self.cfg["env"]["state_observations"]["dof_pos"], device=rl_device
            )
            * self.cfg["env"]["obs_noise"]["dof_pos"],
            torch.ones(
                self.cfg["env"]["state_observations"]["dof_vel"], device=rl_device
            )
            * self.cfg["env"]["obs_noise"]["dof_vel"],
            torch.ones(
                self.cfg["env"]["state_observations"]["last_actions"], device=rl_device
            )
            * self.cfg["env"]["obs_noise"]["last_actions"],
            torch.ones(
                self.cfg["env"]["state_observations"]["gait_sin_indict"],
                device=rl_device,
            )
            * self.cfg["env"]["obs_noise"]["gait_sin_indict"],
            torch.ones(
                self.cfg["env"]["state_observations"]["body_yaw"], device=rl_device
            )
            * self.cfg["env"]["obs_noise"]["body_yaw"],
            torch.ones(
                self.cfg["env"]["state_observations"]["ball_states_p"], device=rl_device
            )
            * self.cfg["env"]["obs_noise"]["ball_states_p"],
            torch.ones(
                self.cfg["env"]["state_observations"]["command"], device=rl_device
            )
            * self.cfg["env"]["obs_noise"]["command"],
        ]

        self.noise_vec = torch.cat(noise_list)

        # history
        self.obs_history = self.cfg["env"]["obs_history"]
        self.history_length = self.cfg["env"]["history_length"]

        # priviledge
        self.obs_privilige = self.cfg["env"]["obs_privilege"]
        if self.obs_privilige:
            privious_obs_dict = self.cfg["env"]["priviledgeStates"]
            privious_obs_dim = 0
            print("privious_obs_dict:", privious_obs_dict)
            for val in privious_obs_dict.values():
                privious_obs_dim += val
            self.privilige_length = privious_obs_dim

        self.cfg["env"]["numActions"] = self.cfg["env"]["actions_num"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        self.cam_range = self.cfg["env"]["pixel_observations"]["cam_range"]
        self.image_width = self.cfg["env"]["pixel_observations"]["width"]
        self.image_height = self.cfg["env"]["pixel_observations"]["height"]
        self.head_cam_pose = self.cfg["env"]["pixel_observations"]["head_cam_pose"]
        self.cam_heading_rad = self.cfg["env"]["pixel_observations"]["cam_heading_rad"]

        fx = (self.image_width / 2) / math.tan(math.radians(self.cam_range / 2))
        fy = (self.image_height / 2) / math.tan(math.radians(self.cam_range / 2))

        self.K_torch = torch.tensor(
            [[fx, 0, self.image_width / 2], [0, fy, self.image_height / 2], [0, 0, 1]],
            dtype=torch.float32,
            device=sim_device,
        )

        def rotation_matrix_x(rad):
            cos_theta = math.cos(rad)
            sin_theta = math.sin(rad)

            rotation_matrix = torch.tensor(
                [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]],
                dtype=torch.float32,
                device=sim_device,
            )
            return rotation_matrix

        self.Rr_c = torch.tensor(
            [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            dtype=torch.float32,
            device=sim_device,
        ) @ rotation_matrix_x(-self.cam_heading_rad)

        # here call _creat_ground and _creat_env
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        obs_space_dict = {}
        obs_space_dict["state_obs"] = spaces.Box(
            np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf
        )

        # rewrite obs space
        if self.pixel_obs:
            obs_space_dict["pixel_obs"] = spaces.Box(
                low=0,
                high=255,
                shape=(self.image_height, self.image_width, 3),
                dtype=np.uint8,
            )

        if self.obs_history:
            obs_space_dict["state_history"] = spaces.Box(
                np.ones(self.num_obs * self.history_length) * -np.Inf,
                np.ones(self.num_obs * self.history_length) * np.Inf,
            )

        if self.obs_privilige:
            obs_space_dict["state_privilige"] = spaces.Box(
                np.ones(self.privilige_length) * -np.Inf,
                np.ones(self.privilige_length) * np.Inf,
            )

        self.obs_space = spaces.Dict(obs_space_dict)

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        self.rand_push = self.cfg["env"]["random_params"]["push"]["enable"]
        self.rand_push_length = int(
            self.cfg["env"]["random_params"]["push"]["interval_s"] / self.dt + 0.5
        )
        self.max_push_vel = self.cfg["env"]["random_params"]["push"]["max_vel"]

        self.drag_ball = self.cfg["env"]["random_params"]["ball_drag"]["enable"]
        self.drag_ball_rand_length = int(
            self.cfg["env"]["random_params"]["ball_drag"]["interval_s"] / self.dt + 0.5
        )

        self.gravity_rand_length = int(
            self.cfg["env"]["random_params"]["gravity"]["interval_s"] / self.dt + 0.5
        )

        self.ball_pos_prob = self.cfg["env"]["random_params"]["ball_reset"]["prob_pos"]
        self.ball_pos_reset = self.cfg["env"]["random_params"]["ball_reset"]["pos"]
        self.ball_vel_prob = self.cfg["env"]["random_params"]["ball_reset"]["prob_vel"]
        self.ball_vel_reset = self.cfg["env"]["random_params"]["ball_reset"]["vel"]

        self._prepare_reward_function()

        self.create_sim_monitor()
        self.create_self_buffers()

        # curricula
        self.command_s = self.cfg["env"]["learn"]["curriculum"]["resample_s"]
        self.curri_resample_length = int(self.command_s / self.dt + 0.5)
        self.curriculum_thresholds = self.cfg["env"]["learn"]["curriculum"][
            "success_threshold"
        ]
        local_range = self.cfg["env"]["learn"]["curriculum"]["local_range"]
        self.curri_local_range = torch.tensor(local_range, device=self.device)
        self.init_curriculum()

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.set_actor_root_state_tensor_indexed()
        # print ("Go1WallKicker init done by gymenv!!")

    def init_curriculum(self):
        # init curriculum
        self.curriculum = RewardThresholdCurriculum(
            device=self.device,
            x_vel=(
                self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"][0],
                self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"][1],
                self.cfg["env"]["randomCommandVelocityRanges"]["num_bins_x"],
            ),
            y_vel=(
                self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"][0],
                self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"][1],
                self.cfg["env"]["randomCommandVelocityRanges"]["num_bins_y"],
            ),
        )

        # print("init curriculum:")
        # print(["env"]["randomCommandVelocityRanges"]["linear_x_init"][0])

        low = torch.tensor(
            [
                self.cfg["env"]["randomCommandVelocityRanges"]["linear_x_init"][0],
                self.cfg["env"]["randomCommandVelocityRanges"]["linear_y_init"][0],
            ],
            device=self.device,
        )
        high = torch.tensor(
            [
                self.cfg["env"]["randomCommandVelocityRanges"]["linear_x_init"][1],
                self.cfg["env"]["randomCommandVelocityRanges"]["linear_y_init"][1],
            ],
            device=self.device,
        )

        self.curriculum.set_to(low, high)

        # self.env_command_bins = np.zeros(self.num_envs, dtype=np.int)
        new_commands, self.env_command_bins = self.curriculum.sample(self.num_envs)
        self.commands = new_commands.to(self.device)

        # setting the smaller commands to zero
        self.commands[:, :2] *= (
            torch.norm(self.commands[:, :2], dim=1) > 0.2
        ).unsqueeze(1)

    def set_viewer(self):
        """Create the viewer."""

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            viewer_prop = gymapi.CameraProperties()
            viewer_prop.use_collision_geometry = True
            viewer_prop.far_plane = 15.0
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, viewer_prop)
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "record_frames"
            )

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim_monitor(self):
        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        # self.base_body_state = self.rigid_body_state.view(
        #     self.num_envs, (self.num_bodies + 1), 13
        # )[:, self.base_index, 0:8]

        self.foot_velocities = self.rigid_body_state.view(
            self.num_envs, (self.num_bodies + 1), 13
        )[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(
            self.num_envs, (self.num_bodies + 1), 13
        )[:, self.feet_indices, 0:3]

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.last_dof_vel = self.dof_vel.clone()

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.rew_pos = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_neg = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        if self.obs_history:
            zero_obs = torch.zeros(
                self.num_obs, dtype=torch.float32, device=self.device
            )
            zero_obs[2] = -1  # default gravity
            self.history_per_begin = torch.tile(zero_obs, (self.history_length,))
            self.history_buffer = torch.tile(self.history_per_begin, (self.num_envs, 1))

        if self.obs_privilige:
            self.privilige_buffer = torch.zeros(
                self.num_envs,
                self.privilige_length,
                device=self.device,
                dtype=torch.float,
            )

    def create_self_buffers(self):
        # initialize some data used later on
        # the differce with monitor is that these are not wrapped gym-state tensors

        self.step_counter = 0

        self.force_tensor = torch.zeros(
            (self.num_envs, self.num_bodies + 1, 3),
            dtype=torch.float32,
            device=self.device,
        )

        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        # default gait
        self.frequencies = (
            torch.ones(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            * self.cfg["env"]["gait_condition"]["frequency"]
        )

        self.phases = self.cfg["env"]["gait_condition"]["phases"]
        self.offsets = self.cfg["env"]["gait_condition"]["offsets"]
        self.bounds = self.cfg["env"]["gait_condition"]["bounds"]
        self.kappa = self.cfg["env"]["gait_condition"]["kappa"]  # from walk these ways

        self.durations = (
            torch.ones(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            * self.cfg["env"]["gait_condition"]["duration"]
        )

        self.desired_contact_states = torch.zeros(
            self.num_envs,
            4,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.gait_indices = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.clock_inputs = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.commands = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.is_first_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )

        self.commands_x = self.commands.view(self.num_envs, 2)[..., 0]
        self.commands_y = self.commands.view(self.num_envs, 2)[..., 1]

        self.ball_lin_vel_xy_world = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.object_local_pos = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.default_dof_pos = torch.zeros_like(
            self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.ball_near_feets = torch.zeros(
            (self.num_envs, 4),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        self.target_reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )

        self.camera_tensor_imgs_buf = torch.zeros(
            [self.num_envs, self.image_width, self.image_width, 3],
            dtype=torch.uint8,
            device=self.device,
            requires_grad=False,
        )

        self.initial_root_states = self.root_states.clone()
        # self.initial_root_states[:] = to_torch(
        #     self.base_init_state, device=self.device, requires_grad=False
        # )
        # self.gravity_vec = to_torch(
        #     get_axis_params(-1.0, self.up_axis_idx), device=self.device
        # ).repeat((self.num_envs, 1))

        self.gravity_vec = to_torch(
            self.cfg["sim"]["gravity"], device=self.device
        ).repeat(
            (self.num_envs, 1)
        )  # TODO: here actually make gravity not a unit vector
        self.gravity_vec /= 9.81
        self.gravity_offset_rand_params = torch.zeros_like(
            self.gravity_vec, device=self.device
        )

        self.lag_buffer = [
            torch.zeros_like(self.dof_pos, device=self.device)
            for i in range(self.cfg["env"]["action_lag_step"] + 1)
        ]

        self.ball_pos = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float, requires_grad=False
        )

        self.ball_p_buffer = [
            torch.zeros_like(self.ball_pos, device=self.device)
            for i in range(self.cfg["env"]["vision_lag_step"] + 1)
        ]

        self.ball_v_buffer = [
            torch.zeros_like(self.ball_pos, device=self.device)
            for i in range(self.cfg["env"]["vision_lag_step"] + 1)
        ]

        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.targets = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_targets = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_last_targets = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.actor_indices_for_reset: List[torch.Tensor] = []

        self.ball_qstar_element = torch.diag(
            torch.tensor([1.0, 1.0, 1.0, -1 / 0.1**2], device=self.device)
        )
        # 0.01, 0.3, 0.3, asset_options_box
        self.goal_qstar_element = torch.diag(
            torch.tensor([0.01**2, 0.15**2, 0.15**2, -1], device=self.device)
        )
        self.proot_cam_element = torch.tensor(self.head_cam_pose, device=self.device)

        self.robot_ball_dis = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        # for PID
        self.v_dog_target = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.omega_dog_local = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

    def create_sim(self):
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        # plane_params.rolling_friction = 0.6
        self.gym.add_ground(self.sim, plane_params)

    def _prepare_rand_params(self):
        self.dof_stiff_rand_params = torch.zeros(
            self.num_envs, self.num_dof, device=self.device, dtype=torch.float
        )
        self.dof_calib_rand_params = torch.zeros(
            self.num_envs, self.num_dof, device=self.device, dtype=torch.float
        )
        self.dof_damping_rand_params = torch.zeros(
            self.num_envs, self.num_dof, device=self.device, dtype=torch.float
        )
        self.payload_rand_params = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.float
        )
        self.friction_rand_params = torch.zeros(
            self.num_envs, 4, device=self.device, dtype=torch.float
        )
        self.restitution_rand_params = torch.zeros(
            self.num_envs, 4, device=self.device, dtype=torch.float
        )
        self.com_rand_params = torch.zeros(self.num_envs, 3, device=self.device)

        self.ball_mass_rand_params = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.float
        )
        self.ball_restitution_rand_params = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.float
        )
        self.ball_drag_rand_params = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../assets"
        )
        asset_file = "urdf/go1/urdf/go1.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = False
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        # asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        a1 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(a1)
        self.num_bodies = self.gym.get_asset_rigid_body_count(a1)
        print("=== num_bodies:", self.num_bodies)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(a1)
        print("=== body_names:", body_names)
        self.dof_names = self.gym.get_asset_dof_names(a1)
        # extremity_name = "SHANK" if asset_options.collapse_fixed_joints else "FOOT"
        extremity_name = "foot"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        knee_names = [s for s in body_names if ("hip" in s) or ("thigh" in s)]
        self.knee_indices = torch.zeros(
            len(knee_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        thigh_names = [s for s in body_names if ("thigh" in s)]
        calf_names = [s for s in body_names if ("calf" in s)]
        self.penalised_contact_indices = torch.zeros(
            len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(a1)
        for i in range(self.num_dof):
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][i] = self.cfg["env"]["control"][
                "stiffness"
            ]  # self.Kp
            dof_props["damping"][i] = self.cfg["env"]["control"]["damping"]  # self.Kd

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_options = gymapi.AssetOptions()
        asset_options.density = 0.1
        asset_ball = self.gym.create_sphere(self.sim, 0.095, asset_options)

        # randomization params, here because some param can only be applied
        # during set up

        if self.do_rand:
            self._prepare_rand_params()
            self.sample_rand_params()

            dog_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(a1)
            ball_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(
                asset_ball
            )

        self.a1_handles = []
        self.envs = []
        if self.add_real_ball:
            self.ball_handles = []

        if self.pixel_obs or self.have_cam_window:
            self.camera_handles = []
            self.camera_tensor_list = []
            self.frame_count = 0

        silver_color = gymapi.Vec3(0.5, 0.5, 0.5)
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            if self.do_rand:
                for s in range(4):  # four legs and feets
                    for b in range(4):  # 5 bodies each side
                        dog_rigid_shape_props[1 + 4 * s + b].restitution = (
                            self.cfg["env"]["random_params"]["restitution"]["range_low"]
                            + (
                                self.cfg["env"]["random_params"]["restitution"][
                                    "range_high"
                                ]
                                - self.cfg["env"]["random_params"]["restitution"][
                                    "range_low"
                                ]
                            )
                            * self.restitution_rand_params[i, s].item()
                        )
                        dog_rigid_shape_props[1 + 4 * s + b].friction = (
                            self.cfg["env"]["random_params"]["friction"]["range_low"]
                            + (
                                self.cfg["env"]["random_params"]["friction"][
                                    "range_high"
                                ]
                                - self.cfg["env"]["random_params"]["friction"][
                                    "range_low"
                                ]
                            )
                            * self.friction_rand_params[i, s].item()
                        )

                self.gym.set_asset_rigid_shape_properties(a1, dog_rigid_shape_props)

            a1_handle = self.gym.create_actor(env_ptr, a1, start_pose, "a1", i, 0, 0)

            if self.do_rand:
                if i == 0:
                    self.default_mass = self.gym.get_actor_rigid_body_properties(
                        env_ptr, a1_handle
                    )[0].mass
                # Rigid
                rigid_props = self.gym.get_actor_rigid_body_properties(
                    env_ptr, a1_handle
                )
                rigid_props[0].mass = (
                    self.default_mass
                    + self.payload_rand_params[i, 0].item()
                    * (
                        self.cfg["env"]["random_params"]["payload"]["range_high"]
                        - self.cfg["env"]["random_params"]["payload"]["range_low"]
                    )
                    + self.cfg["env"]["random_params"]["payload"]["range_low"]
                )
                rigid_props[0].com = gymapi.Vec3(0.0, 0.0, 0)
                rigid_props[0].com = gymapi.Vec3(
                    self.com_rand_params[i, 0]
                    * (
                        self.cfg["env"]["random_params"]["com"]["range_high"]
                        - self.cfg["env"]["random_params"]["com"]["range_low"]
                    )
                    + self.cfg["env"]["random_params"]["com"]["range_low"]
                    + self.cfg["env"]["random_params"]["com"]["x_offset"],
                    self.com_rand_params[i, 1]
                    * (
                        self.cfg["env"]["random_params"]["com"]["range_high"]
                        - self.cfg["env"]["random_params"]["com"]["range_low"]
                    )
                    + self.cfg["env"]["random_params"]["com"]["range_low"],
                    self.com_rand_params[i, 2]
                    * (
                        self.cfg["env"]["random_params"]["com"]["range_high"]
                        - self.cfg["env"]["random_params"]["com"]["range_low"]
                    )
                    + self.cfg["env"]["random_params"]["com"]["range_low"],
                )

                self.gym.set_actor_rigid_body_properties(
                    env_ptr, a1_handle, rigid_props, recomputeInertia=True
                )

                self.gym.set_actor_dof_properties(env_ptr, a1_handle, dof_props)

            self.gym.enable_actor_dof_force_sensors(env_ptr, a1_handle)
            self.envs.append(env_ptr)
            self.a1_handles.append(a1_handle)

            if self.pixel_obs or self.have_cam_window:
                color = gymapi.Vec3(1, 0, 0)
                color_goal = gymapi.Vec3(0, 1, 0)
            else:
                c = 0.7 * np.random.random(3)
                color = gymapi.Vec3(c[0], c[1], c[2])
                color_goal = gymapi.Vec3(c[0], c[1], c[2])

            if self.add_real_ball:
                if self.do_rand:
                    ball_rigid_shape_props[0].restitution = (
                        self.cfg["env"]["random_params"]["ball_restitution"][
                            "range_low"
                        ]
                        + (
                            self.cfg["env"]["random_params"]["ball_restitution"][
                                "range_high"
                            ]
                            - self.cfg["env"]["random_params"]["ball_restitution"][
                                "range_low"
                            ]
                        )
                        * self.ball_restitution_rand_params[i, 0].item()
                    )
                    self.gym.set_asset_rigid_shape_properties(
                        asset_ball, ball_rigid_shape_props
                    )

                ball_handle = self.gym.create_actor(
                    env_ptr,
                    asset_ball,
                    gymapi.Transform(gymapi.Vec3(*self.ball_init_pos)),
                    "ball",
                    i,
                    0b001,
                    1,
                )

                if self.do_rand:
                    this_ball_phy_props = self.gym.get_actor_rigid_body_properties(
                        env_ptr, ball_handle
                    )
                    this_ball_phy_props[0].mass = (
                        self.cfg["env"]["random_params"]["ball_mass"]["range_low"]
                        + (
                            self.cfg["env"]["random_params"]["ball_mass"]["range_high"]
                            - self.cfg["env"]["random_params"]["ball_mass"]["range_low"]
                        )
                        * self.ball_mass_rand_params[i, 0].item()
                    )

                    self.gym.set_actor_rigid_body_properties(
                        env_ptr, ball_handle, this_ball_phy_props
                    )

                self.gym.set_rigid_body_color(
                    env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color
                )
                self.ball_handles.append(ball_handle)

            # gymutil.draw_lines(sphere_geom, self.gym, self.viewer, env_ptr, gymapi.Transform(0,1,2))

            # set color for each go1
            self.gym.reset_actor_materials(
                env_ptr, a1_handle, gymapi.MESH_VISUAL_AND_COLLISION
            )
            self.gym.set_rigid_body_color(
                env_ptr, a1_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color
            )

            # set silver for every other part of go1
            for j in range(1, self.gym.get_actor_rigid_body_count(env_ptr, a1_handle)):
                self.gym.set_rigid_body_color(
                    env_ptr, a1_handle, j, gymapi.MESH_VISUAL_AND_COLLISION, color
                )

            if self.pixel_obs or self.have_cam_window:
                self.add_cam(env_ptr, a1_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.a1_handles[0], feet_names[i]
            )
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.a1_handles[0], knee_names[i]
            )
        for i in range(len(thigh_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.a1_handles[0], thigh_names[i]
            )

        self.base_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.a1_handles[0], "base"
        )

        self.ball_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.ball_handles[0], "ball"
        )

    def sample_rand_params(self):
        self.payload_rand_params = torch.rand((self.num_envs, 1), device=self.device)
        self.com_rand_params = torch.rand((self.num_envs, 3), device=self.device)
        self.friction_rand_params = torch.rand((self.num_envs, 4), device=self.device)
        self.restitution_rand_params = torch.rand(
            (self.num_envs, 4), device=self.device
        )
        self.ball_mass_rand_params = torch.rand((self.num_envs, 1), device=self.device)
        self.ball_restitution_rand_params = torch.rand(
            (self.num_envs, 1), device=self.device
        )

    def add_cam(self, env_ptr, a1_handle):
        camera_properties = gymapi.CameraProperties()
        camera_properties.horizontal_fov = self.cam_range
        camera_properties.enable_tensors = True
        camera_properties.width = self.image_width
        camera_properties.height = self.image_height

        cam_handle = self.gym.create_camera_sensor(env_ptr, camera_properties)
        camera_offset = gymapi.Vec3(
            self.head_cam_pose[0], self.head_cam_pose[1], self.head_cam_pose[2]
        )
        camera_rotation = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 1, 0), self.cam_heading_rad
        )
        body_handle = self.gym.get_actor_rigid_body_handle(env_ptr, a1_handle, 0)

        self.gym.attach_camera_to_body(
            cam_handle,
            env_ptr,
            body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_TRANSFORM,
        )
        self.camera_handles.append(cam_handle)
        cam_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR
        )

        # wrap camera tensor in a pytorch tensor
        torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
        self.camera_tensor_list.append(torch_cam_tensor)

    def wandb_addtional_log(self):
        if self.step_counter % self.wandb_log_period == 0:
            if wandb.run is not None:
                plt.ylabel(
                    "x-command in [{},{}]".format(
                        self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"][0],
                        self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"][1],
                    )
                )
                plt.xlabel(
                    "y-command in [{},{}]".format(
                        self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"][0],
                        self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"][1],
                    )
                )
                wandb.log(
                    {
                        "curriculum": plt.imshow(
                            self.curriculum.weights_shaped.cpu(),
                            cmap="gray",
                            vmin=0.0,
                            vmax=1.0,
                        ).get_figure()
                    }
                )
                total_grid_num = self.curriculum.weights_shaped.numel()
                total_grid_weight = torch.sum(self.curriculum.weights_shaped).item()
                wandb.log(
                    {"curriculum complete rate": total_grid_weight / total_grid_num}
                )

    def pre_physics_step(self, actions):
        self.last_last_targets = self.last_targets.clone()
        self.last_targets = self.targets.clone()
        self.last_dof_vel = (
            self.dof_vel.clone()
        )  # because the vel will be updated in the physics step, so we have to save the last vel before the physics step
        self.last_actions = self.actions.clone()
        self.actions = actions.clone().to(self.device)

        actions_scaled = actions[:, :12] * self.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *= self.hip_addtional_scale

        self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone().to(self.device)]
        self.targets = (
            self.lag_buffer[0]
            + self.default_dof_pos
            + self.cfg["env"]["random_params"]["dof_calib"]["range_low"]
            + (
                self.cfg["env"]["random_params"]["dof_calib"]["range_high"]
                - self.cfg["env"]["random_params"]["dof_calib"]["range_low"]
            )
            * self.dof_calib_rand_params
        )

        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.targets)
        )

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get("actions", None):
            actions = self.dr_randomizations["actions"]["noise_lambda"](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            if self.force_render:
                self.render()

            self.gym.simulate(self.sim)

        # to fix!
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        self.control_steps += 1

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (
            self.reset_buf != 0
        )

        # randomize observations
        if self.dr_randomizations.get("observations", None):
            self.obs_buf = self.dr_randomizations["observations"]["noise_lambda"](
                self.obs_buf
            )

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)
        self.extras["is_firsts"] = self.is_first_buf.to(self.rl_device)

        self.obs_dict["obs"] = {
            "state_obs": torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(
                self.rl_device
            )
        }

        if self.obs_history:
            self.obs_dict["obs"]["state_history"] = torch.clamp(
                self.history_buffer, -self.clip_obs, self.clip_obs
            ).to(self.rl_device)

        if self.obs_privilige:
            self.obs_dict["obs"]["state_privilige"] = self.privilige_buffer.to(
                self.rl_device
            )

        if self.pixel_obs:
            self.obs_dict["obs"]["pixel_obs"] = self.camera_tensor_imgs_buf.to(
                self.rl_device
            )

        # for i in self.obs_dict["obs"].values():
        #     print("i th env obs:", i.shape, i.dtype)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        # self.obs_dict["pixel"] = self.history_images.to(self.rl_device)
        # for i in self.obs_dict["obs"].keys():
        #     print("i th obs:", i)

        return (
            self.obs_dict,
            self.rew_buf.to(self.rl_device),
            self.reset_buf.to(self.rl_device),
            self.extras,
        )

    def post_physics_step(self):
        self.progress_buf += 1

        # the reset is from the previous step
        # because we need the observation of 0 step, if compute_reset -> reset, we will lose the observation of 0 step

        if self.rand_push:
            push_ids = (
                (self.progress_buf % self.rand_push_length == 0)
                .nonzero(as_tuple=False)
                .flatten()
            )

            if len(push_ids) > 0:
                self._push_robots(push_ids)

        self.is_first_buf[:] = False
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(
            -1
        )  # env_ides is [id1 id2 ...]
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            self.is_first_buf[env_ids] = True

        if self.cfg["env"]["random_params"]["ball_reset"]["enable"]:
            self._randomize_ball_state()

        self.set_actor_root_state_tensor_indexed()

        # print("root_states before:", self.root_states)

        self.refresh_self_buffers()
        self._step_contact_targets()

        # print("root_states:", self.root_states)

        # print("=====================================================")
        # print("event: ", self.is_back)
        # print("near wall: ", self.ball_near_wall_now)
        # print("near robot: ", self.ball_near_robot_now)
        # print("wall distance: ", torch.abs(self.ball_pos[:, 0] - self.wall_init_pos[0]))
        # print("rebound times: ", self.rebound_times)

        resample_ids = (
            (self.progress_buf % self.curri_resample_length == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )

        self.compute_reward(self.actions)

        if len(resample_ids) > 0:
            self.resample_commands(resample_ids)

        self.compute_observations()

        if wandb.run is not None and self.wandb_extra_log:
            self.wandb_addtional_log()

        self.compute_reset()

        if self.pixel_obs or self.have_cam_window:
            self.compute_pixels()

        if self.do_rand and self.step_counter % self.gravity_rand_length == 0:
            self.randomize_dof_props()
            if self.cfg["env"]["random_params"]["gravity"]["enable"]:
                self.randomize_gravity()

        if self.drag_ball:
            if self.step_counter % self.drag_ball_rand_length == 0:
                print("=== randomize ball drag")
                self._randomize_ball_drag()
            self._apply_drag_force()

        self.step_counter += 1

    def compute_pixels(self):
        self.frame_count += 1
        if self.device != "cpu":
            self.gym.fetch_results(
                self.sim, True
            )  # True means wait for GPU physics to finish, because we need to access the camera results

        self.gym.step_graphics(self.sim)
        # self.render()
        # self.gym.draw_viewer(self.viewer, self.sim, True)

        # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)

        self.gym.start_access_image_tensors(self.sim)

        # self.camera_tensor_imgs = torch.stack(self.camera_tensor_list,dim=0)

        # self.history_images = torch.cat([camera_tensor_imgs[:,:,:,:3].unsqueeze(1).clone(),self.history_images[:,0:3,:,:,:3]],dim=1)
        self.camera_tensor_imgs_buf = torch.stack(self.camera_tensor_list, dim=0)[
            :, :, :, :3
        ].clone()

        self.gym.end_access_image_tensors(self.sim)

        root_states = self.root_states[:, :]

        if self.have_cam_window:
            if np.mod(self.frame_count, 1) == 0:
                # for i in range(len(self.camera_handles)):
                # up_row = torch.cat([self.history_images[0,0,:,:,:],self.history_images[0,1,:,:,:]],dim=1)
                # low_row = torch.cat([self.history_images[0,2,:,:,:],self.history_images[0,3,:,:,:]],dim=1)
                # whole_picture = torch.cat((up_row, low_row), dim=0)
                # cam_img = whole_picture.cpu().numpy()

                # bboxes_ball = quadric_utils.calc_projected_bbox(self.ball_qstar_element, base_quat, base_pose, self.K_torch, ball_pose)

                pixel_bbox_ball = quadric_utils.convert_bbox_to_int_index(
                    self.bboxes_ball, self.image_width, self.image_height
                )

                pixel_bbox_ball01 = quadric_utils.convert_bbox_to_01(
                    self.bboxes_ball,
                    self.image_width,
                    self.image_height,
                    size_tolerance=1,
                )

                # print("=== ball bbox:",bboxes_ball)
                print("=== pixel ball bbox:", pixel_bbox_ball01)

                # bboxes_goal = quadric_utils.calc_projected_bbox(self.goal_qstar_element, base_quat, base_pose, self.K_torch, goal_pose)

                # print("=== goal bbox:",bboxes_goal)
                # print("=== pixel goal bbox:",pixel_bbox_goal)

                cam_img = (
                    self.camera_tensor_imgs_buf[self.num_envs - 1, :, :, :]
                    .cpu()
                    .numpy()
                )
                cam_img = quadric_utils.add_bbox_on_numpy_img(
                    cam_img,
                    pixel_bbox_ball[self.num_envs - 1, 0].item(),
                    pixel_bbox_ball[self.num_envs - 1, 1].item(),
                    pixel_bbox_ball[self.num_envs - 1, 2].item(),
                    pixel_bbox_ball[self.num_envs - 1, 3].item(),
                )

                # pixel_bbox_goal = quadric_utils.convert_bbox_to_img_coord(
                #     self.bboxes_goal, self.image_width, self.image_height
                # )
                # cam_img = quadric_utils.add_bbox_on_numpy_img(
                #     cam_img,
                #     pixel_bbox_goal[self.num_envs - 1, 0].item(),
                #     pixel_bbox_goal[self.num_envs - 1, 1].item(),
                #     pixel_bbox_goal[self.num_envs - 1, 2].item(),
                #     pixel_bbox_goal[self.num_envs - 1, 3].item(),
                #     box_color=(0, 255, 0),
                # )

                self.ax.imshow(cam_img)
                plt.draw()
                plt.pause(0.01)
                self.ax.cla()

    def refresh_self_buffers(self):
        # refresh
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # raw data, rigid body state need extra refresh
        # self.base_body_state = self.rigid_body_state.view(
        #     self.num_envs, (self.num_bodies + 1), 13
        # )[:, self.base_index, 0:8]
        self.foot_velocities = self.rigid_body_state.view(
            self.num_envs, (self.num_bodies + 1), 13
        )[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(
            self.num_envs, (self.num_bodies + 1), 13
        )[:, self.feet_indices, 0:3]

        # state data
        self.base_quat = self.root_states[::2, 3:7]
        self.base_pos = self.root_states[::2, 0:3]
        self.ball_pos = self.root_states[1::2, 0:3]

        self.ball_near_feets = (
            torch.norm(self.foot_positions - self.ball_pos.unsqueeze(1), dim=2) < 0.2
        )  # ball radius is 0.1

        # print("feet in body frame:")

        # print(
        #     quat_rotate_inverse(
        #         self.base_quat, self.foot_positions[:, 0, :] - self.base_pos
        #     )
        # )

        # print("=== ball near feet:", self.ball_near_feets)

        self.robot_ball_dis = torch.norm(
            self.ball_pos - self.base_pos, dim=1, keepdim=False
        )

        self.ball_lin_vel_xy_world = self.root_states[1::2, 7:9]

        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.contact_state = (self.contact_forces[:, self.feet_indices, 2] > 1.0).view(
            self.num_envs, -1
        ) * 1.0
        self.base_lin_vel = self.root_states[::2, 7:10]
        self.true_object_local_pos = quat_rotate_inverse(
            self.base_quat, self.root_states[1::2, 0:3] - self.root_states[0::2, 0:3]
        )
        self.true_object_local_pos[:, 2] = 0.0 * torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.true_object_local_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[1::2, 7:10]
        )

        self.ball_p_buffer = self.ball_p_buffer[1:] + [
            self.true_object_local_pos.clone().to(self.device)
        ]

        self.ball_v_buffer = self.ball_v_buffer[1:] + [
            self.true_object_local_vel.clone().to(self.device)
        ]

        self.object_local_pos = self.simulate_ball_pos_delay(
            self.ball_p_buffer[0], self.object_local_pos
        )
        self.object_lin_vel = self.root_states[1::2, 7:10]

        # self.feet_air_time += self.dt

    def compute_reward(self, actions):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        extra_info = {}
        episode_cumulative = {}
        self.rew_buf[:] = 0.0
        self.rew_pos[:] = 0.0
        self.rew_neg[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            # print("name:", name, "scale:", self.reward_scales[name])
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if name == "raibert_heuristic_PID":
                self.negtive_shaper = rew
            elif name == "tracking_lin_vel_PID":
                self.positive_shaper = rew
            elif torch.sum(rew) >= 0:
                self.rew_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_neg += rew
            episode_cumulative[name] = rew

            # if name == 'tracking_lin_vel':
            #     print("max velocity",torch.max(self.reward_functions[i]()).item(), self.reward_scales[name])

            if name in [
                "tracking_contacts_shaped_force",
                "tracking_contacts_shaped_vel",
            ]:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew

        if self.postive_reward_ji22:
            self.rew_buf[:] = self.rew_pos[:] * torch.exp(
                self.rew_neg[:] / self.sigma_rew_neg
            )
        else:
            self.rew_buf = self.rew_pos + self.rew_neg

        episode_cumulative["total_unshaped"] = self.rew_buf.clone()

        if self.PID_shaper:
            self.rew_buf[:] = torch.exp(self.negtive_shaper / self.sigma_rew_neg) * (
                self.rew_buf + self.positive_shaper
            )

        extra_info["episode_cumulative"] = episode_cumulative
        self.extras.update(extra_info)

    def compute_reset(self):
        reset = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.0
        reset = reset | torch.any(
            torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1.0, dim=1
        )

        # too far away
        reset = reset | (self.robot_ball_dis > self.robot_ball_max)

        reset = reset | torch.any(
            self.ball_lin_vel_xy_world > self.ball_speed_max, dim=1
        )

        # reset = reset | self.ball_in_goal_now
        time_out = self.progress_buf >= self.max_episode_length - 1
        reset = reset | time_out

        self.reset_buf[:] = reset

        # self.target_reset_buf was set in the reward function

    def compute_observations(self):
        # self.compute_PID_commands()
        root_states = self.root_states[:, :]
        # if self.step_counter < 50 * 7:
        #     self.commands[:, 0] = -1.0
        #     self.commands[:, 1] = 0.0
        # else:
        #     self.commands[:, 0] = 1.0
        #     self.commands[:, 1] = 0.0
        # print("commands:", self.commands)
        commands = self.commands
        dof_pos = self.dof_pos
        default_dof_pos = self.default_dof_pos
        dof_vel = self.dof_vel
        gravity_vec = self.gravity_vec
        actions = self.actions
        lin_vel_scale = self.lin_vel_scale
        ang_vel_scale = self.ang_vel_scale
        dof_pos_scale = self.dof_pos_scale
        dof_vel_scale = self.dof_vel_scale

        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
        yaw = wrap_to_pi(heading)

        base_quat = root_states[::2, 3:7]
        base_pose = root_states[::2, 0:3]
        base_lin_vel = (
            quat_rotate_inverse(base_quat, root_states[::2, 7:10]) * lin_vel_scale
        )
        base_ang_vel = (
            quat_rotate_inverse(base_quat, root_states[::2, 10:13]) * ang_vel_scale
        )
        projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)
        dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

        commands_scaled = commands * lin_vel_scale

        ball_states_p = self.object_local_pos
        ball_states_v = quat_rotate_inverse(
            self.base_quat, self.root_states[1::2, 7:10]
        )

        pw_ball = root_states[1::2, 0:3] - root_states[0::2, 0:3]
        vw_ball = root_states[1::2, 7:10]

        cat_list = []
        # check dict have key
        if "base_lin_vel" in self.cfg["env"]["state_observations"]:
            cat_list.append(base_lin_vel)
        if "base_ang_vel" in self.cfg["env"]["state_observations"]:
            cat_list.append(base_ang_vel)
        if "projected_gravity" in self.cfg["env"]["state_observations"]:
            cat_list.append(projected_gravity)

        if "dof_pos" in self.cfg["env"]["state_observations"]:
            # print("dof_pos:",dof_pos_scaled)
            cat_list.append(dof_pos_scaled)
        if "dof_vel" in self.cfg["env"]["state_observations"]:
            # print("dof_vel:",dof_vel * dof_vel_scale)
            cat_list.append(dof_vel * dof_vel_scale)
        if "last_actions" in self.cfg["env"]["state_observations"]:
            # print("last_actions:",actions)
            cat_list.append(actions)

        if "gait_sin_indict" in self.cfg["env"]["state_observations"]:
            cat_list.append(self.clock_inputs)

        if "body_yaw" in self.cfg["env"]["state_observations"]:
            cat_list.append(yaw)

        if "ball_states_p" in self.cfg["env"]["state_observations"]:
            cat_list.append(ball_states_p)

        if "command" in self.cfg["env"]["state_observations"]:
            cat_list.append(commands_scaled)

        obs = torch.cat(cat_list, dim=-1)

        if self.add_noise:
            obs += (2 * torch.rand_like(obs) - 1) * self.noise_vec

        self.obs_buf[:] = obs
        # print("ball pos:", obs[:, -5:-2])
        # print("obs_buf:", obs[:, -2:])
        # print("ball_vel", self.object_lin_vel[:, :2])
        # print("body yaw", obs[:, -6])
        if self.obs_history:
            self.history_buffer[:] = torch.cat(
                (self.history_buffer[:, self.num_obs :], obs), dim=1
            )

        # self.command_squence[self.plot_step] = self.commands[0, 0].item()
        # self.velocity_squence[self.plot_step] = self.ball_lin_vel_xy_world[0, 0].item()

        # self.plot_step += 1
        # if self.plot_step >= 14 * 50:
        #     # plot command together with velocity
        #     plt.plot(self.command_squence)
        #     plt.plot(self.velocity_squence)

        #     plt.show()

        if self.obs_privilige:
            # base_lin_vel: 3
            # base_ang_vel: 3

            # ball_states_v: 3 # can be in world frame, because have yaw
            # ball_states_p: 3

            # dof_stiff: 12
            # dof_damp: 12
            # dof_calib: 12
            # payload: 1
            # com: 3
            # friction: 4
            # restitution: 4

            # ball_mass: 1
            # ball_restitution: 1

            # ball_drag: 1
            priv_list = []
            if "base_lin_vel" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(base_lin_vel)
            if "base_ang_vel" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(base_ang_vel)
            if "base_height" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(base_pose[:, 2:3])

            if "ball_states_p_0" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.ball_p_buffer[0])
            if "ball_states_v_0" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.ball_v_buffer[0])

            # 0,4,9,14
            if "ball_states_p_3" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.ball_p_buffer[3])
            if "ball_states_v_3" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.ball_v_buffer[3])

            if "ball_states_p_6" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.ball_p_buffer[6])
            if "ball_states_v_6" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.ball_v_buffer[6])

            if "ball_states_p_1" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.ball_p_buffer[1])
            if "ball_states_v_1" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.ball_v_buffer[1])

            if "ball_states_p_2" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.ball_p_buffer[2])
            if "ball_states_v_2" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.ball_v_buffer[2])

            if "dof_stiff" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.dof_stiff_rand_params)

            if "dof_damp" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.dof_damping_rand_params)

            if "dof_calib" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.dof_calib_rand_params)

            if "payload" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.payload_rand_params)

            if "com" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.com_rand_params)

            if "friction" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.friction_rand_params)

            if "restitution" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.restitution_rand_params)

            if "ball_mass" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.ball_mass_rand_params)

            if "ball_restitution" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.ball_restitution_rand_params)

            if "gravity_offset" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.gravity_offset_rand_params)

            if "ball_drag" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.ball_drag_rand_params)

            if not self.cfg["env"]["empty_privilege"]:
                self.privilige_buffer[:] = torch.cat(priv_list, dim=-1)
                # print("privilege buffer:", self.privilige_buffer)

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, which will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # reward containers
        from isaacgymenvs.tasks.go1func.dribble_rewards import RewardTerms

        self.reward_container = RewardTerms(self)

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                print(
                    f"Reward {key} has nonzero coefficient {scale}, multiplying by dt={self.dt}"
                )
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        self.PID_shaper = False
        for name, scale in self.reward_scales.items():
            if name == "raibert_heuristic_PID":
                self.PID_shaper = True
            if name == "termination":
                continue

            if not hasattr(self.reward_container, "_reward_" + name):
                ImportError(
                    f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!"
                )
            else:
                self.reward_names.append(name)
                self.reward_functions.append(
                    getattr(self.reward_container, "_reward_" + name)
                )

        # reward episode sums
        # self.episode_sums = {
        #     name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        #     for name in self.reward_scales.keys()}
        # self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
        #                                          requires_grad=False)
        # self.episode_sums_eval = {
        #     name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        #     for name in self.reward_scales.keys()}
        # self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
        #                                               requires_grad=False)
        self.command_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in list(self.reward_scales.keys())
            + [
                "lin_vel_raw",
                "ang_vel_raw",
                "lin_vel_residual",
                "ang_vel_residual",
                "ep_timesteps",
            ]
        }

    def compute_PID_commands(self):
        delta_v = self.ball_lin_vel_xy_world - self.commands[:, 0:2]
        p_dog_target = (
            self.ball_pos[:, :2]
            + delta_v * self.reward_params["raibert_heuristic_PID"]["k1"]
        )
        self.v_dog_target[:, 0:2] = (
            p_dog_target - self.base_pos[:, :2]
        ) * self.reward_params["raibert_heuristic_PID"]["k2"] + (
            self.ball_lin_vel_xy_world - self.base_lin_vel[:, :2]
        ) * self.reward_params[
            "raibert_heuristic_PID"
        ][
            "k3"
        ]
        self.v_dog_local = quat_rotate_inverse(self.base_quat, self.v_dog_target)

        # forward = quat_apply(self.base_quat, self.forward_vec)
        # robot_heading = torch.atan2(forward[:, 1], forward[:, 0])
        # command_heading = torch.atan2(self.v_dog_local[:, 1], self.v_dog_local[:, 0])
        # self.omega_dog_local[:] = torch.clip(
        #     0.5 * wrap_to_pi(command_heading - robot_heading), -1.0, 1.0
        # )

    def reset(self):
        """Is called only once when environment starts to provide the first observations.
        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """

        # self.compute_observations()
        # self.compute_reset()
        # self.compute_reward(self.actions)

        self.obs_dict["obs"] = {
            "state_obs": torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(
                self.rl_device
            )
        }

        if self.pixel_obs:
            self.compute_pixels()
            self.obs_dict["obs"]["pixel_obs"] = self.camera_tensor_imgs_buf.to(
                self.rl_device
            )

        if self.obs_history:
            self.obs_dict["obs"]["state_history"] = torch.clamp(
                self.history_buffer, -self.clip_obs, self.clip_obs
            ).to(self.rl_device)

        if self.obs_privilige:
            self.obs_dict["obs"]["state_privilige"] = self.privilige_buffer.to(
                self.rl_device
            )

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict

    def reset_idx(self, env_ids):
        num_ids = len(env_ids)
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            print("Randomizing...")
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(
            0.5, 1.5, (len(env_ids), self.num_dof), device=self.device
        )
        velocities = torch_rand_float(
            -0.1, 0.1, (len(env_ids), self.num_dof), device=self.device
        )

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        actor_indices = torch.cat([env_ids * 2, env_ids * 2 + 1])

        # print("whole reset index", actor_indices)

        # copy initial robot and ball states
        self.root_states[env_ids * 2] = self.initial_root_states[env_ids * 2].clone()
        self.root_states[env_ids * 2 + 1] = self.initial_root_states[
            env_ids * 2 + 1
        ].clone()

        # orientation randomization
        random_yaw_angle = (
            2
            * (
                torch.rand(
                    len(env_ids),
                    3,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
                - 0.5
            )
            * torch.tensor([0, 0, 3.14], device=self.device)
        )
        self.root_states[env_ids * 2, 3:7] = quat_from_euler_xyz(
            random_yaw_angle[:, 0], random_yaw_angle[:, 1], random_yaw_angle[:, 2]
        )
        self.root_states[env_ids * 2, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )

        self.root_states[env_ids * 2 + 1, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )

        # reset the ball pose, jump by 2, because a1,ball,a1,ball
        self.root_states[env_ids * 2 + 1, 0] = (
            torch.ones([num_ids], device=self.device)
            * (self.ball_init_pos[0] - self.ball_rand_pos_range[0] / 2.0)
            + torch.rand([num_ids], device=self.device) * self.ball_rand_pos_range[0]
        )
        self.root_states[env_ids * 2 + 1, 1] = (
            torch.ones([num_ids], device=self.device)
            * (self.ball_init_pos[1] - self.ball_rand_pos_range[1] / 2.0)
            + torch.rand([num_ids], device=self.device) * self.ball_rand_pos_range[1]
        )
        self.root_states[env_ids * 2 + 1, 2] = (
            torch.ones([num_ids], device=self.device)
            * (self.ball_init_pos[2] - self.ball_rand_pos_range[2] / 2.0)
            + torch.rand([num_ids], device=self.device) * self.ball_rand_pos_range[2]
        )

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.deferred_set_actor_root_state_tensor_indexed(actor_indices)

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32 * 2),
            len(env_ids_int32),
        )

        self.gait_indices[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.last_targets[env_ids] = 0.0
        self.last_last_targets[env_ids] = 0.0
        self.object_local_pos[env_ids] = 0.0

        if self.obs_history:
            self.history_buffer[env_ids, :] = torch.tile(
                self.history_per_begin, (len(env_ids), 1)
            )

        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0

        for i in range(len(self.ball_p_buffer)):
            self.ball_p_buffer[i][env_ids, :] = 0

        for i in range(len(self.ball_v_buffer)):
            self.ball_v_buffer[i][env_ids, :] = 0

        # self.commands_x[env_ids] = torch_rand_float(
        #     self.command_x_range[0],
        #     self.command_x_range[1],
        #     (len(env_ids), 1),
        #     device=self.device,
        # ).squeeze()
        # self.commands_y[env_ids] = torch_rand_float(
        #     self.command_y_range[0],
        #     self.command_y_range[1],
        #     (len(env_ids), 1),
        #     device=self.device,
        # ).squeeze()

    def deferred_set_actor_root_state_tensor_indexed(
        self, obj_indices: List[torch.Tensor]
    ) -> None:
        self.actor_indices_for_reset.append(obj_indices)

    def set_actor_root_state_tensor_indexed(self) -> None:
        object_indices: List[torch.Tensor] = self.actor_indices_for_reset
        if not object_indices:
            # nothing to set
            return
        # print("index list:", object_indices)
        unique_object_indices = torch.unique(torch.cat(object_indices).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(unique_object_indices),
            len(unique_object_indices),
        )

        self.actor_indices_for_reset = []

    def resample_commands(self, env_ids):
        old_bins = self.env_command_bins[env_ids]
        # print("old_bins_is",old_bins)

        task_rewards, success_thresholds = [], []
        for key in [
            "dribbling_ball_vel",
            "tracking_contacts_shaped_force",
            "tracking_contacts_shaped_vel",
        ]:
            if key in self.command_sums.keys():
                task_rewards.append(
                    self.command_sums[key][env_ids] / self.curri_resample_length
                )  # because scale have additional dt
                success_thresholds.append(
                    self.curriculum_thresholds[key] * self.reward_scales[key]
                )
                # print("========================")

                # print("max command sums = ",key,torch.max(self.command_sums[key][env_ids]).item() / self.curri_resample_length)
                # print("command standard = ",key,self.curriculum_thresholds[key]* self.reward_scales[key])
                # print("resample length", self.curri_resample_length)
                # print("scales",key,self.reward_scales[key])

        self.curriculum.update(
            old_bins,
            task_rewards,
            success_thresholds,
            local_range=self.curri_local_range,
        )

        new_commands, new_bin_inds = self.curriculum.sample(batch_size=len(env_ids))
        self.env_command_bins[env_ids] = new_bin_inds
        self.commands[env_ids] = new_commands[:, :].to(self.device)

        # setting the smaller commands to zero
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > 0.2
        ).unsqueeze(1)

        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.0

    def _step_contact_targets(self):
        self.gait_indices = torch.remainder(
            self.gait_indices + self.dt * self.frequencies, 1.0
        )

        foot_indices = [
            self.gait_indices + self.phases + self.offsets + self.bounds,
            self.gait_indices + self.offsets,
            self.gait_indices + self.bounds,
            self.gait_indices + self.phases,
        ]

        self.foot_indices = torch.remainder(
            torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0
        )

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < self.durations
            swing_idxs = torch.remainder(idxs, 1) > self.durations

            # print(stance_idxs)
            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (
                0.5 / self.durations[stance_idxs]
            )

            idxs[swing_idxs] = 0.5 + (
                torch.remainder(idxs[swing_idxs], 1) - self.durations[swing_idxs]
            ) * (0.5 / (1 - self.durations[swing_idxs]))

        # if self.cfg.commands.durations_warp_clock_inputs:

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        smoothing_cdf_start = torch.distributions.normal.Normal(
            0, self.kappa
        ).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = smoothing_cdf_start(
            torch.remainder(foot_indices[0], 1.0)
        ) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)
        ) + smoothing_cdf_start(
            torch.remainder(foot_indices[0], 1.0) - 1
        ) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)
        )
        smoothing_multiplier_FR = smoothing_cdf_start(
            torch.remainder(foot_indices[1], 1.0)
        ) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)
        ) + smoothing_cdf_start(
            torch.remainder(foot_indices[1], 1.0) - 1
        ) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)
        )
        smoothing_multiplier_RL = smoothing_cdf_start(
            torch.remainder(foot_indices[2], 1.0)
        ) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)
        ) + smoothing_cdf_start(
            torch.remainder(foot_indices[2], 1.0) - 1
        ) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)
        )
        smoothing_multiplier_RR = smoothing_cdf_start(
            torch.remainder(foot_indices[3], 1.0)
        ) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)
        ) + smoothing_cdf_start(
            torch.remainder(foot_indices[3], 1.0) - 1
        ) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)
        )

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR

    def _randomize_ball_drag(self):
        self.ball_drag_rand_params[:, :] = torch_rand_float(
            self.cfg["env"]["random_params"]["ball_drag"]["range_low"],
            self.cfg["env"]["random_params"]["ball_drag"]["range_high"],
            (self.num_envs, 1),
            device=self.device,
        )

    def _apply_drag_force(self):
        self.force_tensor[:, self.num_bodies, :2] = (
            -self.ball_drag_rand_params
            # * torch.square(self.ball_lin_vel_xy_world[:, :2])
            # * torch.sign(self.ball_lin_vel_xy_world[:, :2])
            * self.ball_lin_vel_xy_world[:, :2]
        )
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.force_tensor),
            None,
            gymapi.ENV_SPACE,
        )

    def _randomize_ball_state(self):
        reset_ball_pos_mark = np.random.choice(
            [True, False],
            self.num_envs,
            p=[self.ball_pos_prob, 1 - self.ball_pos_prob],
        )
        reset_ball_pos_env_ids = torch.tensor(
            np.array(np.nonzero(reset_ball_pos_mark)), device=self.device
        ).flatten()  # reset_ball_pos_mark.nonzero(as_tuple=False).flatten()
        ball_pos_env_ids = (
            reset_ball_pos_env_ids.to(device=self.device) * 2 + 1
        )  # ball index
        # reset_ball_pos_env_ids_int32 = ball_pos_env_ids.to(dtype=torch.int32)
        self.root_states[ball_pos_env_ids, 0:3] += (
            2
            * (
                torch.rand(
                    len(ball_pos_env_ids),
                    3,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
                - 0.5
            )
            * torch.tensor(self.ball_pos_reset, device=self.device, requires_grad=False)
        )

        reset_ball_vel_mark = np.random.choice(
            [True, False],
            self.num_envs,
            p=[self.ball_vel_prob, 1 - self.ball_vel_prob],
        )
        reset_ball_vel_env_ids = torch.tensor(
            np.array(np.nonzero(reset_ball_vel_mark)), device=self.device
        ).flatten()
        ball_vel_env_ids = reset_ball_vel_env_ids.to(device=self.device) * 2 + 1
        self.root_states[ball_vel_env_ids, 7:10] = (
            2
            * (
                torch.rand(
                    len(ball_vel_env_ids),
                    3,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
                - 0.5
            )
            * torch.tensor(self.ball_pos_reset, device=self.device, requires_grad=False)
        )

        # reset_ball_vel_env_ids_int32 = ball_vel_env_ids.to(dtype=torch.int32)
        self.deferred_set_actor_root_state_tensor_indexed(ball_pos_env_ids)
        self.deferred_set_actor_root_state_tensor_indexed(ball_vel_env_ids)

    def _push_robots(self, env_ids):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        # print("pushing robots", env_ids)

        self.root_states[env_ids * 2, 7:9] += torch_rand_float(
            -self.max_push_vel, self.max_push_vel, (len(env_ids), 2), device=self.device
        )  # lin vel x/y
        # print("root states", self.root_states[:, 7:9])
        self.deferred_set_actor_root_state_tensor_indexed(env_ids * 2)
        # # for debug
        # self.root_states[env_ids * 2 + 1, 7] = 0.0
        # self.root_states[env_ids * 2 + 1, 8] = 1.0
        # self.deferred_set_actor_root_state_tensor_indexed(env_ids * 2 + 1)

    def simulate_ball_pos_delay(self, new_ball_pos, last_ball_pos):
        receive_mark = np.random.choice(
            [True, False],
            self.num_envs,
            p=[
                self.vision_receive_prob,
                1 - self.vision_receive_prob,
            ],
        )
        last_ball_pos[receive_mark, :] = new_ball_pos[receive_mark, :]

        return last_ball_pos

    def randomize_dof_props(self):
        print("=== randomize properties of the environment")
        self.dof_stiff_rand_params = torch.rand(
            (self.num_envs, self.num_dof), device=self.device
        )
        self.dof_damping_rand_params = torch.rand(
            (self.num_envs, self.num_dof), device=self.device
        )
        self.dof_calib_rand_params = torch.rand(
            (self.num_envs, self.num_dof), device=self.device
        )

        for i in range(self.num_envs):
            actor_handle = self.a1_handles[i]
            env_handle = self.envs[i]
            # DOF
            dof_props = self.gym.get_actor_dof_properties(env_handle, actor_handle)
            # print("==== dof_props:", dof_props)
            for s in range(self.num_dof):
                dof_props["stiffness"][s] = (
                    self.cfg["env"]["random_params"]["stiffness"]["range_low"]
                    + (
                        self.cfg["env"]["random_params"]["stiffness"]["range_high"]
                        - self.cfg["env"]["random_params"]["stiffness"]["range_low"]
                    )
                    * self.dof_stiff_rand_params[i, s].item()
                )
                dof_props["damping"][s] = (
                    self.cfg["env"]["random_params"]["damping"]["range_low"]
                    + (
                        self.cfg["env"]["random_params"]["damping"]["range_high"]
                        - self.cfg["env"]["random_params"]["damping"]["range_low"]
                    )
                    * self.dof_damping_rand_params[i, s].item()
                )
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

    def randomize_gravity(self):
        print("=== randomize gravity of the environment")
        prop = self.gym.get_sim_params(self.sim)
        gravity = [0.0, 0.0, -9.8]
        gravity[0] = (
            self.cfg["env"]["random_params"]["gravity"]["range_low"]
            + (
                self.cfg["env"]["random_params"]["gravity"]["range_high"]
                - self.cfg["env"]["random_params"]["gravity"]["range_low"]
            )
            * torch.rand(1).item()
        )
        gravity[1] = (
            self.cfg["env"]["random_params"]["gravity"]["range_low"]
            + (
                self.cfg["env"]["random_params"]["gravity"]["range_high"]
                - self.cfg["env"]["random_params"]["gravity"]["range_low"]
            )
            * torch.rand(1).item()
        )
        gravity[2] += (
            self.cfg["env"]["random_params"]["gravity"]["range_low"]
            + (
                self.cfg["env"]["random_params"]["gravity"]["range_high"]
                - self.cfg["env"]["random_params"]["gravity"]["range_low"]
            )
            * torch.rand(1).item()
        )
        prop.gravity = gymapi.Vec3(*gravity)
        self.gym.set_sim_params(self.sim, prop)  # save for sim
        self.gravity_vec[:, :] = torch.tensor(gravity, device=self.device)

        self.gravity_offset_rand_params = self.gravity_vec[:, :] - torch.tensor(
            [0.0, 0.0, -9.8], device=self.device
        )
        self.gravity_vec /= 9.81
        # save for monitor
