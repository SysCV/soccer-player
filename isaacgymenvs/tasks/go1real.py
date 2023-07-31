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

from typing import Tuple, Dict, Any

import copy
import sys
import time
import math

import glob
import pickle as pkl
import lcm
import sys

from go1_gym_deploy.utils.deployment_runner import DeploymentRunner
from go1_gym_deploy.envs.lcm_agent import LCMAgent
from go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go1_gym_deploy.utils.command_profile import *

import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

class Go1Real(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        # ==================== for real robot =====================
        print("Initializing the real robot .....................")
        self.timestep = 0
        self.se = StateEstimator(lc)

        max_vel=1.0
        max_yaw_vel=1.0
        control_dt = 0.02
        self.dt = control_dt

        command_profile = RCControllerProfile(dt=control_dt, state_estimator=self.se, x_scale=max_vel, y_scale=0.6, yaw_scale=max_yaw_vel)
        self.command_profile = command_profile
        hardware_agent = LCMAgent(cfg, self.se, command_profile, device = rl_device)
        self.agents = {}
        self.control_agent_name = "hardware_closed_loop"
        self.agents["hardware_closed_loop"] = hardware_agent
        self.se.spin()
        self.button_states = np.zeros(4)

        for agent_name in self.agents.keys():
            obs = self.agents[agent_name].reset()
            control_obs = obs

        control_obs = self.calibrate(wait=True)


        # ============================

        self.cfg = cfg
        # cam pic
        self.save_cam = self.cfg["task"]["save_cam_pic"]
        self.add_fake_ball = self.cfg["task"]["fake_ball"]
        self.real = self.cfg["env"]["real_robot"]


        
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.hip_addtional_scale = self.cfg["env"]["control"]["hipAddtionalScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]

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
        self.cfg["env"]["numObservations"] = total_obs_dim 
    
        self.cfg["env"]["numActions"] = self.cfg["env"]["act_num"] 

        # don't need to create sim here
        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        config = self.cfg
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = rl_device

        # Rendering
        # if training in a headless mode
        self.headless = headless

        enable_camera_sensors = config.get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_environments = config["env"]["numEnvs"]
        self.num_agents = config["env"].get("numAgents", 1)  # used for multi-agent environments

        self.num_observations = config["env"].get("numObservations", 0)
        self.num_states = config["env"].get("numStates", 0)

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)

        self.num_actions = config["env"]["numActions"]
        self.control_freq_inv = config["env"].get("controlFrequencyInv", 1)

        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.clip_obs = config["env"].get("clipObservations", np.Inf)
        self.clip_actions = config["env"].get("clipActions", np.Inf)

        # Total number of training frames since the beginning of the experiment.
        # We get this information from the learning algorithm rather than tracking ourselves.
        # The learning algorithm tracks the total number of frames since the beginning of training and accounts for
        # experiments restart/resumes. This means this number can be > 0 right after initialization if we resume the
        # experiment.
        self.total_train_env_frames: int = 0

        # number of control steps
        self.control_steps: int = 0

        self.render_fps: int = config["env"].get("renderFPS", -1)
        self.last_frame_time: float = 0.0
        self.obs_dict = {}



        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long) # in real, never reset
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        self.obs_buf = control_obs

    def create_sim(self):
        pass

    def calibrate(self, wait=True, low=False):
        # first, if the robot is not in nominal pose, move slowly to the nominal pose
        print("Calibrating the robot !!!!")
        if hasattr(self.agents["hardware_closed_loop"], "get_obs"):
            agent = self.agents["hardware_closed_loop"]
            agent.get_obs()
            joint_pos = agent.dof_pos
            if low:
                final_goal = np.array([0., 0.3, -0.7,
                                        0., 0.3, -0.7,
                                        0., 0.3, -0.7,
                                        0., 0.3, -0.7,])
            else:
                final_goal = np.zeros(12)
            nominal_joint_pos = agent.default_dof_pos

            print(f"About to calibrate; the robot will stand [Press R2 to calibrate]")
            while wait:
                self.button_states = self.command_profile.get_buttons()
                if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                    break

            cal_action = np.zeros((agent.num_envs, agent.num_actions))
            target_sequence = []
            target = joint_pos - nominal_joint_pos
            while np.max(np.abs(target - final_goal)) > 0.01:
                target -= np.clip((target - final_goal), -0.05, 0.05)
                target_sequence += [copy.deepcopy(target)]
            for target in target_sequence:
                next_target = target
                if isinstance(agent.cfg, dict):
                    hip_reduction = 1.0
                    action_scale = agent.cfg["env"]["control"]["actionScale"]

                next_target[[0, 3, 6, 9]] /= hip_reduction
                next_target = next_target / action_scale
                cal_action[:, 0:12] = next_target
                agent.step(torch.from_numpy(cal_action))
                agent.get_obs()
                time.sleep(0.05)

            print("Starting pose calibrated [Press R2 to start controller]")
            while True:
                self.button_states = self.command_profile.get_buttons()
                if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                    break

            for agent_name in self.agents.keys():
                obs = self.agents[agent_name].reset()
                if agent_name == "hardware_closed_loop":
                    control_obs = obs

        return control_obs
    

    def pre_physics_step(self, actions):
        self.time = time.time()
        self.actions = actions.clone().to(self.device)
        # self.actions[:,0:3] = 0.

        # actions_scaled = self.actions[:, :12] * self.action_scale
        # actions_scaled[:, [0, 3, 6, 9]] *= self.hip_addtional_scale
        # targets = actions_scaled
        self.agents["hardware_closed_loop"].publish_action(self.actions, hard_reset=False) # default pos will be added inside the agent

        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.timestep % 100 == 0: print(f'frq: {1 / (time.time() - self.time)} Hz');
        self.time = time.time()

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)


        # compute observations, rewards, resets, ...
        self.post_physics_step()

        self.control_steps += 1


        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)


        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def post_physics_step(self):

        self.compute_observations()
        self.compute_reward(self.actions)

        rpy = self.agents[self.control_agent_name].se.get_rpy()
        if abs(rpy[0]) > 1.6 or abs(rpy[1]) > 1.6:
            self.calibrate(wait=False, low=True)


        if self.command_profile.state_estimator.right_lower_right_switch_pressed:
            control_obs = self.calibrate(wait=False)
            time.sleep(1)
            self.command_profile.state_estimator.right_lower_right_switch_pressed = False
            # self.button_states = self.command_profile.get_buttons()
            while not self.command_profile.state_estimator.right_lower_right_switch_pressed:
                time.sleep(0.01)
                # self.button_states = self.command_profile.get_buttons()
            self.command_profile.state_estimator.right_lower_right_switch_pressed = False


    def compute_reward(self, actions):
        self.rew_buf[0] = 0

    def compute_observations(self):
        obs = self.agents["hardware_closed_loop"].get_obs()




        gravity_vec = obs[:,0:3]
        commands = obs[:,3:6]
        dof_pos = obs[:,6:18]
        dof_vel = obs[:,18:30]
        actions = obs[:,30:42] # actions are not used in real robot
        contact_states = obs[:,42:46]

        # print("=== gravity_vec: ", gravity_vec)
        # print("=== contact_states: ", contact_states)

        lin_vel_scale = self.lin_vel_scale
        ang_vel_scale = self.ang_vel_scale
        dof_pos_scale = self.dof_pos_scale
        dof_vel_scale = self.dof_vel_scale


        projected_gravity = gravity_vec
        dof_pos_scaled = dof_pos * dof_pos_scale
        dof_vel_scaled = dof_vel * dof_vel_scale
        commands_scaled = commands*torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale], requires_grad=False, device=commands.device)


        obs = torch.cat((
            #base_lin_vel,
            #base_ang_vel,
            projected_gravity,
            commands_scaled,
            dof_pos_scaled,
            dof_vel_scaled,
            actions,
            contact_states
        ), dim=-1)

        self.obs_buf[:] = obs


    def reset_idx(self, env_ids):
        pass

