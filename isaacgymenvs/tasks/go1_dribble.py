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

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgym import gymutil

from isaacgymenvs.tasks.base.vec_task import VecTask
import isaacgymenvs.tasks.go1func.quadric_utils as quadric_utils

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

        self.cfg = cfg
        # cam pic
        self.have_cam_window = self.cfg["env"]["cameraSensorPlt"]
        self.pixel_obs = self.cfg["env"]["pixel_observations"]["enable"]
        # print("pixel_obs:", self.pixel_obs)
        if self.have_cam_window:
            _, self.ax = plt.subplots()
            plt.axis("off")
        self.add_real_ball = self.cfg["task"]["target_ball"]

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
        self.robot_x_range = self.cfg["env"]["terminateCondition"]["robot_x_range"]
        self.ball_x_range = self.cfg["env"]["terminateCondition"]["ball_x_range"]
        self.ball_y_range = self.cfg["env"]["terminateCondition"]["ball_y_range"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        # self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        # self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        # self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

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

        self._prepare_reward_function()

        self.create_sim_monitor()
        self.create_self_buffers()

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.set_actor_root_state_tensor_indexed()
        # print ("Go1WallKicker init done by gymenv!!")

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

        self.default_dof_pos = torch.zeros_like(
            self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False
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
        self.initial_root_states[:] = to_torch(
            self.base_init_state, device=self.device, requires_grad=False
        )
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))

        self.lag_buffer = [
            torch.zeros_like(self.dof_pos, device=self.device)
            for i in range(self.cfg["env"]["action_lag_step"] + 1)
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

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../assets"
        )
        asset_file = "urdf/go1/urdf/go1.urdf"
        # asset_path = os.path.join(asset_root, asset_file)
        # asset_root = os.path.dirname(asset_path)
        # asset_file = os.path.basename(asset_path)

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
        asset_ball = self.gym.create_sphere(self.sim, 0.1, asset_options)

        self.a1_handles = []
        self.envs = []
        if self.add_real_ball:
            self.ball_handles = []

        self.goal_handles = []
        self.wall_handles = []

        if self.pixel_obs or self.have_cam_window:
            self.camera_handles = []
            self.camera_tensor_list = []
            self.frame_count = 0

        silver_color = gymapi.Vec3(0.5, 0.5, 0.5)
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            a1_handle = self.gym.create_actor(
                env_ptr, a1, start_pose, "a1", i, 0b010, 0
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
                ball_handle = self.gym.create_actor(
                    env_ptr,
                    asset_ball,
                    gymapi.Transform(gymapi.Vec3(*self.ball_init_pos)),
                    "ball",
                    i,
                    0b001,
                    1,
                )

                this_ball_props = self.gym.get_actor_rigid_shape_properties(
                    env_ptr, ball_handle
                )
                this_ball_props[0].rolling_friction = 0.1
                this_ball_props[0].restitution = 0.9
                self.gym.set_actor_rigid_shape_properties(
                    env_ptr, ball_handle, this_ball_props
                )

                this_ball_phy_props = self.gym.get_actor_rigid_body_properties(
                    env_ptr, ball_handle
                )
                this_ball_phy_props[0].mass = self.ball_mass
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
        self.targets = self.lag_buffer[0] + self.default_dof_pos

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

        goal_env_ids = self.target_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # print(goal_env_ids)
        if len(goal_env_ids) > 0:
            self.reset_only_target(goal_env_ids)

        self.is_first_buf[:] = False
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(
            -1
        )  # env_ides is [id1 id2 ...]
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            self.is_first_buf[env_ids] = True

        self.set_actor_root_state_tensor_indexed()

        self.refresh_self_buffers()
        self._step_contact_targets()

        # print("=====================================================")
        # print("event: ", self.is_back)
        # print("near wall: ", self.ball_near_wall_now)
        # print("near robot: ", self.ball_near_robot_now)
        # print("wall distance: ", torch.abs(self.ball_pos[:, 0] - self.wall_init_pos[0]))
        # print("rebound times: ", self.rebound_times)

        self.compute_observations()
        self.compute_reset()
        self.compute_reward(self.actions)

        if self.pixel_obs or self.have_cam_window:
            self.compute_pixels()

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

        # state data
        self.base_quat = self.root_states[::2, 3:7]
        self.base_pos = self.root_states[::2, 0:3]
        self.ball_pos = self.root_states[1::2, 0:3]
        self.ball_lin_vel_xy_world = self.root_states[1::2, 7:9]

        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.contact_state = (self.contact_forces[:, self.feet_indices, 2] > 1.0).view(
            self.num_envs, -1
        ) * 1.0
        self.base_lin_vel = self.root_states[::2, 7:10]
        self.object_local_pos = quat_rotate_inverse(
            self.base_quat, self.root_states[1::2, 0:3] - self.root_states[0::2, 0:3]
        )
        self.object_local_pos[:, 2] = 0.0 * torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
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
            if torch.sum(rew) >= 0:
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

        # print(episode_cumulative)
        extra_info["episode_cumulative"] = episode_cumulative
        self.extras.update(extra_info)

    def compute_reset(self):
        reset = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.0
        reset = reset | torch.any(
            torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1.0, dim=1
        )

        # reset = reset | self.ball_in_goal_now
        time_out = self.progress_buf >= self.max_episode_length - 1
        reset = reset | time_out

        self.reset_buf[:] = reset

        # self.target_reset_buf was set in the reward function

    def compute_observations(self):
        root_states = self.root_states[:, :]
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

        pw_ball = root_states[1::2, 0:3]

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

        self.obs_buf[:] = obs

        if self.obs_history:
            self.history_buffer[:] = torch.cat(
                (self.history_buffer[:, self.num_obs :], obs), dim=1
            )

        if self.obs_privilige:
            priv_list = []
            if "base_lin_vel" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(base_lin_vel)
            if "base_ang_vel" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(base_ang_vel)

            if "ball_states_p" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(ball_states_p)
            if "ball_states_v" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(ball_states_v)

            if "base_pose" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(base_pose)
            if "base_quat" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(base_quat)
            if "ball_event" in self.cfg["env"]["priviledgeStates"]:
                priv_list.append(self.is_back.unsqueeze(1).to(torch.float32))

            if not self.cfg["env"]["empty_privilege"]:
                self.privilige_buffer[:] = torch.cat(priv_list, dim=-1)

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
        for name, scale in self.reward_scales.items():
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

        # jump by 2, because a1,ball,a1,ball
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

        if self.obs_history:
            self.history_buffer[env_ids, :] = torch.tile(
                self.history_per_begin, (len(env_ids), 1)
            )

        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0

        self.commands_x[env_ids] = torch_rand_float(
            -1, 1, (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands_y[env_ids] = torch_rand_float(
            -1, 1, (len(env_ids), 1), device=self.device
        ).squeeze()

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
