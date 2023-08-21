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

# ======== Asset info unitree a1: ========
# Got 17 bodies, 16 joints, and 12 DOFs
# 
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
#
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
# 
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
#
# ======== Asset info unitree a1: ========


import wandb
import numpy as np
from matplotlib import pyplot as plt
import os
import torch

from gym import spaces

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.curricula.curriculum_torch import RewardThresholdCurriculum

from typing import Tuple, Dict, Any, List

import sys
import time
import math


class Go1(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.wandb_extra_log = self.cfg["env"]["wandb_extra_log"]
        self.wandb_log_period = self.cfg["env"]["wandb_extra_log_period"]
        self.random_frec = self.cfg["env"]["randomization_freq"]

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.hip_addtional_scale = self.cfg["env"]["control"]["hipAddtionalScale"]

        self.phases =  self.cfg["env"]["gait_condition"]["phases"]
        self.offsets =  self.cfg["env"]["gait_condition"]["offsets"]
        self.bounds =  self.cfg["env"]["gait_condition"]["bounds"]
        self.kappa = self.cfg["env"]["gait_condition"]["kappa"] # from walk these ways


        self.add_noise = self.cfg["env"]["add_noise"]
        noise_list = [
            torch.ones(self.cfg["env"]["obs_num"]["gravity"], device=rl_device) * self.cfg["env"]["obs_noise"]["gravity"],
            torch.ones(self.cfg["env"]["obs_num"]["command"], device=rl_device) * self.cfg["env"]["obs_noise"]["command"],
            torch.ones(self.cfg["env"]["obs_num"]["dof_p"], device=rl_device) * self.cfg["env"]["obs_noise"]["dof_p"],
            torch.ones(self.cfg["env"]["obs_num"]["dof_v"], device=rl_device) * self.cfg["env"]["obs_noise"]["dof_v"],
            torch.ones(self.cfg["env"]["obs_num"]["pre_act"], device=rl_device) * self.cfg["env"]["obs_noise"]["pre_act"],
            torch.ones(self.cfg["env"]["obs_num"]["gait_sin_indict"], device=rl_device) * self.cfg["env"]["obs_noise"]["gait_sin_indict"],
        ]

        self.noise_vec = torch.cat(noise_list)


        # reward scales
        self.postive_reward_ji22 = self.cfg["env"]["rewards"]["only_positive_rewards_ji22_style"]
        self.sigma_rew_neg = self.cfg["env"]["rewards"]["sigma_rew_neg"]
        self.reward_scales = self.cfg["env"]["rewards"]["rewardScales"]
        self.reward_params = self.cfg["env"]["rewards"]["rewardParams"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

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

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        obs_space_dict = self.cfg["env"]["obs_num"]
        total_obs_dim = 0
        for val in obs_space_dict.values():
            total_obs_dim += val

        self.obs_privilige = self.cfg["env"]["obs_privilege"]
        if self.obs_privilige:
            privious_obs_dict = self.cfg["env"]["privilege_obs_num"]
            privious_obs_dim = 0
            for val in privious_obs_dict.values():
                privious_obs_dim += val
        self.privilige_length = privious_obs_dim

        self.obs_history = self.cfg["env"]["obs_history"]
        self.history_length = self.cfg["env"]["history_length"]
        
        self.cfg["env"]["numObservations"] = total_obs_dim 
    
        self.cfg["env"]["numActions"] = self.cfg["env"]["act_num"] 


        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # rewrite obs space
        obs_space_dict = {}
        obs_space_dict["state_obs"] = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)

        if self.obs_history:
            obs_space_dict["state_history"] = spaces.Box(np.ones(self.num_obs*self.history_length) * -np.Inf, np.ones(self.num_obs*self.history_length) * np.Inf)
            
        if self.obs_privilige:
            obs_space_dict["state_privilige"] = spaces.Box(np.ones(self.privilige_length) * -1., np.ones(self.privilige_length) * 1.)
        
        self.obs_space = spaces.Dict(obs_space_dict)

        self._prepare_reward_function()

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]



        # curricula
        self.command_s = self.cfg["env"]["learn"]["curriculum"]["resample_s"]
        self.curri_resample_length = int(self.command_s/self.dt + 0.5)
        self.curriculum_thresholds = self.cfg["env"]["learn"]["curriculum"]["success_threshold"]

        self.rand_push = self.cfg["env"]["random_params"]["push"]["enable"]
        self.rand_push_length = int(self.cfg["env"]["random_params"]["push"]["interval_s"]/self.dt + 0.5)
        self.max_push_vel = self.cfg["env"]["random_params"]["push"]["max_vel"]



        local_range = self.cfg["env"]["learn"]["curriculum"]["local_range"]
        self.curri_local_range=torch.tensor(
                                      local_range, device=self.device)

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.get_wrapped_tensor()

        self.init_curriculum()

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.set_actor_root_state_tensor_indexed()

    def init_curriculum(self):
        # init curriculum
        self.curriculum = RewardThresholdCurriculum(device=self.device,
                x_vel=(self.cfg["env"]["randomCommandVelocityRanges"]           ["linear_x"][0],
                        self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"][1],
                        self.cfg["env"]["randomCommandVelocityRanges"]["num_bins_x"]),
                y_vel=(self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"][0],
                        self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"][1],
                        self.cfg["env"]["randomCommandVelocityRanges"]["num_bins_y"]),
                yaw_vel=(self.cfg["env"]["randomCommandVelocityRanges"]["yaw"][0],
                        self.cfg["env"]["randomCommandVelocityRanges"]["yaw"][1],
                        self.cfg["env"]["randomCommandVelocityRanges"]["num_bins_yaw"]))
        
        # print("init curriculum:")
        # print(["env"]["randomCommandVelocityRanges"]["linear_x_init"][0])

        low = torch.tensor([
            self.cfg["env"]["randomCommandVelocityRanges"]["linear_x_init"][0],
            self.cfg["env"]["randomCommandVelocityRanges"]["linear_y_init"][0],
            self.cfg["env"]["randomCommandVelocityRanges"]["yaw_init"][0]], device=self.device)
        high = torch.tensor([
            self.cfg["env"]["randomCommandVelocityRanges"]["linear_x_init"][1],
            self.cfg["env"]["randomCommandVelocityRanges"]["linear_y_init"][1],
            self.cfg["env"]["randomCommandVelocityRanges"]["yaw_init"][1]], device=self.device)

        self.curriculum.set_to(low,high)

        # self.env_command_bins = np.zeros(self.num_envs, dtype=np.int)
        new_commands, self.env_command_bins = self.curriculum.sample(self.num_envs)
        self.commands = new_commands.to(self.device)



    def get_wrapped_tensor(self):
        # default gait
        self.frequencies = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) * 3.0

        self.durations = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) * 0.5

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.last_dof_vel = self.dof_vel.clone()
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs * self.num_bodies, :]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,self.feet_indices,7:10]

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        if self.obs_history:
            assert self.num_obs == 46, "total_obs_dim should be 43"
            zero_obs = torch.zeros(self.num_obs, dtype=torch.float32, device=self.device)
            zero_obs[2] = -1 # default gravity
            self.history_per_begin = torch.tile(zero_obs, (self.history_length,))
            self.history_buffer = torch.tile(self.history_per_begin, (self.num_envs,1))
    

        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg["env"]["action_lag_step"]+1)]

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.targets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_targets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_targets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.feet_air_time = torch.zeros(self.num_envs, 4 ,dtype=torch.float, device=self.device, requires_grad=False)
        self.contact_state = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device, requires_grad=False)

        self.rew_pos = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_neg = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                 requires_grad=False)


        self.step_counter = 0
        self.actor_indices_for_reset: List[torch.Tensor] = []



    def prepare_rand_params(self):
        self.dof_stiff_rand_params = torch.zeros(self.num_envs, self.num_dof,device=self.device, dtype=torch.float)
        self.payload_rand_params = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float)
        self.friction_rand_params = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float)
        self.restitution_rand_params = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float)

        self.privilige_buffer = torch.zeros(self.num_envs, self.privilige_length, device=self.device, dtype=torch.float)


    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def set_viewer(self):
        """Create the viewer."""

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            viewer_prop = gymapi.CameraProperties()
            viewer_prop.use_collision_geometry = True
            viewer_prop.far_plane = 30.
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, viewer_prop)
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "record_frames")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/go1/urdf/go1.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        a1 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(a1)
        self.num_bodies = self.gym.get_asset_rigid_body_count(a1)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(a1)
        self.dof_names = self.gym.get_asset_dof_names(a1)
        # print("=== body_names: ", body_names)
        # extremity_name = "SHANK" if asset_options.collapse_fixed_joints else "FOOT"
        extremity_name = "foot"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "thigh" in s or "hip" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(a1)
        # print("=== dof_props: ", dof_props)
        # print("=== dof_props_type: ", type(dof_props))
        dof_props['driveMode'][0:self.num_dof] = gymapi.DOF_MODE_POS
        dof_props['stiffness'][0:self.num_dof] = self.cfg["env"]["control"]["stiffness"] #self.Kp
        dof_props['damping'][0:self.num_dof] = self.cfg["env"]["control"]["damping"] #self.Kd
        # print("=== dof_props: ", dof_props)

        self.default_stiffness = self.cfg["env"]["control"]["stiffness"]


        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)


        self.a1_handles = []
        self.envs = []

        # randomization params, here because some param can only be applied
        # during set up
        self.prepare_rand_params()

        self.payload_rand_params = torch.rand((self.num_envs, 1), device=self.device)  


        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            a1_handle = self.gym.create_actor(env_ptr, a1, start_pose, "a1", i, 0, 0)
            if i == 0:
                self.default_mass = self.gym.get_actor_rigid_body_properties(env_ptr, a1_handle)[0].mass
            # Rigid
            rigid_props = self.gym.get_actor_rigid_body_properties(env_ptr, a1_handle)
            rigid_props[0].mass = self.default_mass + self.payload_rand_params[i, 0].item() * (self.cfg["env"]["random_params"]["payload"]["range_high"] - self.cfg["env"]["random_params"]["payload"]["range_low"]) + self.cfg["env"]["random_params"]["payload"]["range_low"]
            # rigid_props[0].com = gymapi.Vec3(
            #     self.com_rand_params[i, 0] * (self.cfg["env"]["random_params"]["com"]["range_high"]- self.cfg["env"]["random_params"]["com"]["range_low"]) + self.cfg["env"]["random_params"]["com"]["range_low"],
            #     self.com_rand_params[i, 1] * (self.cfg["env"]["random_params"]["com"]["range_high"]- self.cfg["env"]["random_params"]["com"]["range_low"]) + self.cfg["env"]["random_params"]["com"]["range_low"],
            #     self.com_rand_params[i, 2] * (self.cfg["env"]["random_params"]["com"]["range_high"]- self.cfg["env"]["random_params"]["com"]["range_low"]) + self.cfg["env"]["random_params"]["com"]["range_low"])
            self.gym.set_actor_rigid_body_properties(env_ptr, a1_handle, rigid_props, recomputeInertia=True)



            self.gym.set_actor_dof_properties(env_ptr, a1_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, a1_handle)
            self.envs.append(env_ptr)
            self.a1_handles.append(a1_handle)

            c = 0.5 * np.random.random(3)
            color = gymapi.Vec3(c[0], c[1], c[2])


            self.gym.set_rigid_body_color(env_ptr, a1_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)



        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], knee_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "base")
        self.default_mass = self.gym.get_actor_rigid_body_properties(self.envs[0], self.a1_handles[0])[self.base_index].mass

    def pre_physics_step(self, actions):
        self.last_last_targets = self.last_targets.clone()
        self.last_targets = self.targets.clone()
        self.last_dof_vel = self.dof_vel.clone()
        self.actions = actions.clone().to(self.device)

        actions_scaled = actions[:, :12] * self.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *= self.hip_addtional_scale

        self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
        self.targets = self.lag_buffer[0] + self.default_dof_pos

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.targets))

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        self.control_steps += 1

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)
        
        self.obs_dict["obs"] = {"state_obs":torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)}

        if self.obs_history:
            self.obs_dict["obs"]["state_history"] = torch.clamp(self.history_buffer, -self.clip_obs, self.clip_obs).to(self.rl_device)

        if self.obs_privilige:
            self.obs_dict["obs"]["state_privilige"] = self.privilige_buffer.to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def post_physics_step(self):
        self.progress_buf += 1 # this is a tensor for each env


        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1) # env_ides is [id1 id2 ...]

        resample_ids = (self.progress_buf % self.curri_resample_length == 0).nonzero(as_tuple=False).flatten()

        if self.rand_push:
            push_ids =  (self.progress_buf % self.rand_push_length == 0).nonzero(as_tuple=False).flatten()

            if len(push_ids) > 0:
                self._push_robots(push_ids)

        if len(resample_ids) > 0:
            self.resample_commands(resample_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.set_actor_root_state_tensor_indexed()

        self.refresh_buffers()
        self._step_contact_targets()

        if wandb.run is not None and self.wandb_extra_log:
            self.wandb_addtional_log()
        self.compute_reset()
        self.compute_observations()
        self.compute_reward(self.actions)
        if self.step_counter % self.random_frec == 0:
           self.randomize_props()
        self.step_counter += 1

    def _push_robots(self, env_ids):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """


        self.root_states[env_ids, 7:9] += torch_rand_float(-self.max_push_vel, self.max_push_vel, (len(env_ids), 2),
                                                          device=self.device)  # lin vel x/y
        self.deferred_set_actor_root_state_tensor_indexed(env_ids)

    def compute_reward(self, actions):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        extra_info = {}
        episode_cumulative = {}
        self.rew_buf[:] = 0.
        self.rew_pos[:] = 0.
        self.rew_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if torch.sum(rew) >= 0:
                self.rew_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_neg += rew
            episode_cumulative[name] = rew

            # if name == 'tracking_lin_vel':
            #     print("max velocity",torch.max(self.reward_functions[i]()).item(), self.reward_scales[name])
                

            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
        

        if self.postive_reward_ji22:
            self.rew_buf[:] = self.rew_pos[:] * torch.exp(self.rew_neg[:] / self.sigma_rew_neg)
        else:
            self.rew_buf = self.rew_pos + self.rew_neg
        extra_info["episode_cumulative"] = episode_cumulative
        self.extras.update(extra_info)

    def wandb_addtional_log(self):
        if self.step_counter % self.wandb_log_period == 0:
            if wandb.run is not None:

                plt.ylabel("velocity-command in [{},{}]".format(
                    self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"][0],
                    self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"][1]))
                plt.xlabel("angular-command in [{},{}]".format(
                    self.cfg["env"]["randomCommandVelocityRanges"]["yaw"][0],
                    self.cfg["env"]["randomCommandVelocityRanges"]["yaw"][1]
                ))
                wandb.log({"curriculum": plt.imshow(torch.mean(self.curriculum.weights_shaped, axis=1).cpu(), cmap='gray',vmin=0.,vmax=1.).get_figure()})


    def compute_observations(self):

        lin_vel_scale = self.lin_vel_scale
        ang_vel_scale = self.ang_vel_scale
        dof_pos_scale = self.dof_pos_scale
        dof_vel_scale = self.dof_vel_scale



        self.contact_state = (self.contact_forces[:, self.feet_indices, 2] > 1.).view(self.num_envs,
                -1) * 1.0
        

        self.feet_air_time += self.dt

        dof_pos_scaled = (self.dof_pos - self.default_dof_pos) * dof_pos_scale

        commands_scaled = self.commands*torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale], requires_grad=False, device=self.commands.device)


        obs = torch.cat((
            self.projected_gravity,
            commands_scaled,
            dof_pos_scaled,
            self.dof_vel * dof_vel_scale,
            self.actions,
            self.clock_inputs,
        ), dim=-1)

        if self.add_noise:
            obs += (2 * torch.rand_like(obs) - 1) * self.noise_vec

        self.obs_buf[:] = obs

        if self.obs_history:
            self.history_buffer[:] = torch.cat((obs, self.history_buffer[:, self.num_obs:]), dim=1)

        if self.obs_privilige:
            self.privilige_buffer = torch.cat([
                self.base_lin_vel * lin_vel_scale, 
                self.base_ang_vel * ang_vel_scale,
                self.dof_stiff_rand_params, self.payload_rand_params, 
                self.friction_rand_params, 
                self.restitution_rand_params], dim= 1)


    def refresh_buffers(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)


        self.base_pos = self.root_states[:, :3]
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

    def compute_reset(self):
        # reset agents
        reset = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1., dim=1)
        time_out = self.progress_buf >= self.max_episode_length - 1
        # reward for time-outs
        reset = reset | time_out
        
        self.reset_buf[:] = reset


    def randomize_props(self):
        print("=== randomize properties of the environment")
        self.dof_stiff_rand_params = torch.rand((self.num_envs, self.num_dof), device=self.device)  


        self.friction_rand_params = torch.rand((self.num_envs, 4), device=self.device)  
        self.restitution_rand_params = torch.rand((self.num_envs, 4), device=self.device) 
        
        for i in range(self.num_envs):
            actor_handle = self.a1_handles[i]
            env_handle = self.envs[i]
            # DOF
            dof_props = self.gym.get_actor_dof_properties(env_handle, actor_handle)
            # print("==== dof_props:", dof_props)
            for s in range(self.num_dof):
                dof_props["stiffness"][s] = self.cfg["env"]["random_params"]["stiffness"]["range_low"] + (self.cfg["env"]["random_params"]["stiffness"]["range_high"] - self.cfg["env"]["random_params"]["stiffness"]["range_low"]) * self.dof_stiff_rand_params[i, s].item()
            self.gym.set_actor_dof_properties(env_handle,actor_handle, dof_props)




            # Shape
            shape_props_list = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            for s in range(4): # four legs and feets
                shape_props_list[s + 1].friction = self.friction_rand_params[i, s].item() * (self.cfg["env"]["random_params"]["friction"]["range_high"] - self.cfg["env"]["random_params"]["friction"]["range_low"]) + self.cfg["env"]["random_params"]["friction"]["range_low"]
                shape_props_list[s + 1].restitution = self.restitution_rand_params[i, s].item() * (self.cfg["env"]["random_params"]["restitution"]["range_high"] - self.cfg["env"]["random_params"]["restitution"]["range_low"]) + self.cfg["env"]["random_params"]["restitution"]["range_low"]


    def resample_commands(self, env_ids):
        old_bins = self.env_command_bins[env_ids]
        # print("old_bins_is",old_bins)

        task_rewards, success_thresholds = [], []
        for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force",
                    "tracking_contacts_shaped_vel"]:
            if key in self.command_sums.keys():
                task_rewards.append(self.command_sums[key][env_ids] / self.curri_resample_length) # because scale have additional dt
                success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])
                # print("========================")

                # print("max command sums = ",key,torch.max(self.command_sums[key][env_ids]).item() / self.curri_resample_length)
                # print("command standard = ",key,self.curriculum_thresholds[key]* self.reward_scales[key])
                # print("resample length", self.curri_resample_length)
                # print("scales",key,self.reward_scales[key])

        self.curriculum.update(old_bins, task_rewards, success_thresholds,
                                local_range=self.curri_local_range)

        new_commands, new_bin_inds = self.curriculum.sample(batch_size=len(env_ids))
        self.env_command_bins[env_ids] = new_bin_inds
        self.commands[env_ids] = new_commands[:, :].to(self.device)

        # setting the smaller commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.
        


    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        # if self.randomize:
        #     self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids,:] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids,:] = velocities
        self.last_dof_vel[env_ids,:] = velocities



        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.root_states[env_ids, :] = self.initial_root_states[env_ids, :]

        self.deferred_set_actor_root_state_tensor_indexed(env_ids)


        self.gym.set_dof_state_tensor_indexed(self.sim,
                                               gymtorch.unwrap_tensor(self.dof_state),
                                               gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gait_indices[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.last_targets[env_ids] = 0.
        self.last_last_targets[env_ids] = 0.

        self.feet_air_time[env_ids,:] = 0. 
        self.contact_state[env_ids,:] = 0.
        
        if self.obs_history:
            self.history_buffer[env_ids,:] = torch.tile(self.history_per_begin, (len(env_ids), 1))
        
        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0

        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.

    def reset(self):
        """Is called only once when environment starts to provide the first observations.
        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """

        self.obs_buf[:,2] = -1

        self.obs_dict["obs"] = {"state_obs":torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)}

        if self.obs_history:
            self.obs_dict["obs"]["state_history"] = torch.clamp(self.history_buffer, -self.clip_obs, self.clip_obs).to(self.rl_device)

        if self.obs_privilige:
            self.obs_dict["obs"]["state_privilige"] = self.privilige_buffer.to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict



    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # reward containers
        from isaacgymenvs.tasks.go1func.walk_rewards import RewardTerms
        self.reward_container = RewardTerms(self)

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue

            if not hasattr(self.reward_container, '_reward_' + name):
                print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))

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
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}

    def _step_contact_targets(self):
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * self.frequencies, 1.0)


        foot_indices = [self.gait_indices + self.phases + self.offsets + self.bounds,
                            self.gait_indices + self.offsets,
                            self.gait_indices + self.bounds,
                            self.gait_indices + self.phases]
        
        # have bug??? ==================
        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < self.durations
            swing_idxs = torch.remainder(idxs, 1) > self.durations


            # print(stance_idxs)
            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / self.durations[stance_idxs])


            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - self.durations[swing_idxs]) * (
                        0.5 / (1 - self.durations[swing_idxs]))

        # if self.cfg.commands.durations_warp_clock_inputs:

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        # self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
        # self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
        # self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
        # self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

        # self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
        # self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
        # self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
        # self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

        # von mises distribution
        
        smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                self.kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR

    def deferred_set_actor_root_state_tensor_indexed(self, obj_indices: List[torch.Tensor]) -> None:
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
