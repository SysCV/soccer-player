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

from isaacgymenvs.go1_deploy.lcm_agent import LCMAgent
from isaacgymenvs.go1_deploy.utils.cheetah_state_estimator import StateEstimator
from isaacgymenvs.go1_deploy.utils.command_profile import *

import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")


class Go1Real(VecTask):
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
        self.time = time.time()
        self.cfg = cfg
        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"][
            "linear_x"
        ]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"][
            "linear_y"
        ]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        self.dt = control_dt = self.cfg["sim"]["dt"]

        self.phases = self.cfg["env"]["gait_condition"]["phases"]
        self.offsets = self.cfg["env"]["gait_condition"]["offsets"]
        self.bounds = self.cfg["env"]["gait_condition"]["bounds"]

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.hip_addtional_scale = self.cfg["env"]["control"]["hipAddtionalScale"]

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        obs_space_dict = self.cfg["env"]["obs_num"]
        total_obs_dim = 0
        for val in obs_space_dict.values():
            total_obs_dim += val
        self.cfg["env"]["numObservations"] = total_obs_dim

        self.cfg["env"]["numActions"] = self.cfg["env"]["act_num"]

        self.obs_history = self.cfg["env"]["obs_history"]
        self.history_length = self.cfg["env"]["history_length"]

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
                print(
                    "GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline."
                )
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = rl_device

        self.num_environments = config["env"]["numEnvs"]
        self.num_agents = config["env"].get(
            "numAgents", 1
        )  # used for multi-agent environments

        self.num_observations = config["env"].get("numObservations", 0)
        self.num_states = config["env"].get("numStates", 0)

        # rewrite obs space
        obs_space_dict = {}
        obs_space_dict["state_obs"] = spaces.Box(
            np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf
        )

        if self.obs_history:
            obs_space_dict["state_history"] = spaces.Box(
                np.ones(self.num_obs * self.history_length) * -np.Inf,
                np.ones(self.num_obs * self.history_length) * np.Inf,
            )

        self.obs_space = spaces.Dict(obs_space_dict)

        self.state_space = spaces.Box(
            np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf
        )

        self.num_actions = config["env"]["numActions"]
        self.control_freq_inv = config["env"].get("controlFrequencyInv", 1)

        self.act_space = spaces.Box(
            np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0
        )

        self.clip_obs = config["env"].get("clipObservations", np.Inf)
        self.clip_actions = config["env"].get("clipActions", np.Inf)

        self.total_train_env_frames: int = 0

        self.init_monitors()

        # ==================== for real robot =====================
        print("Initializing the real robot .....................")
        self.timestep = 0
        self.se = StateEstimator(lc)

        max_x_vel = self.command_x_range[1]
        max_y_vel = self.command_y_range[1]
        max_yaw_vel = self.command_yaw_range[1]

        command_profile = RCControllerProfile(
            dt=control_dt,
            state_estimator=self.se,
            x_scale=max_x_vel,
            y_scale=max_y_vel,
            yaw_scale=max_yaw_vel,
        )
        self.command_profile = command_profile
        hardware_agent = LCMAgent(cfg, self.se, command_profile, device=rl_device)
        self.agents = {}
        self.control_agent_name = "hardware_closed_loop"
        self.agents["hardware_closed_loop"] = hardware_agent
        self.se.spin()
        self.button_states = np.zeros(4)

        for agent_name in self.agents.keys():
            obs = self.agents[agent_name].reset()

        # ============================

        self.calibrate()

    def init_monitors(self):
        # number of control steps
        self.control_steps: int = 0

        self.render_fps: int = self.cfg["env"].get("renderFPS", -1)
        self.last_frame_time: float = 0.0
        self.obs_dict = {}

        # is empty, just for compatibility
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )

        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float
        )
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )  # in real, never reset
        self.timeout_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.lag_buffer = [
            torch.zeros(self.num_envs, 12, device=self.device, dtype=torch.float)
            for i in range(self.cfg["env"]["action_lag_step"] + 1)
        ]

        self.commands = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        joint_names = [
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ]
        # this is in right order

        self.default_dof_pos = (
            torch.tensor(
                [self.cfg["env"]["defaultJointAngles"][name] for name in joint_names]
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
            .to(self.device)
        )

        self.dof_pos = self.default_dof_pos.clone()
        self.dof_vel = torch.zeros(
            self.num_envs,
            12,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.gait_indices = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.clock_inputs = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=torch.float
        )

        if self.obs_history:
            assert self.num_obs == 46, "total_obs_dim should be 46"
            zero_obs = torch.zeros(
                self.num_obs, dtype=torch.float32, device=self.device
            )
            zero_obs[2] = -1  # default gravity
            self.history_per_begin = torch.tile(zero_obs, (self.history_length,))
            self.history_buffer = torch.tile(self.history_per_begin, (self.num_envs, 1))

        self.extras = {}

    def create_sim(self):
        pass

    def calibrate(self, wait=True, low=False):
        # first, if the robot is not in nominal pose, move slowly to the nominal pose
        print("Calibrating the robot to stand pose!!!!")
        if hasattr(self.agents["hardware_closed_loop"], "get_obs"):
            agent = self.agents["hardware_closed_loop"]
            agent.get_obs()
            joint_pos = agent.dof_pos
            if low:
                final_goal = np.array(
                    [
                        0.0,
                        0.3,
                        -0.7,
                        0.0,
                        0.3,
                        -0.7,
                        0.0,
                        0.3,
                        -0.7,
                        0.0,
                        0.3,
                        -0.7,
                    ]
                )
            else:
                final_goal = np.zeros(12)
            nominal_joint_pos = agent.default_dof_pos

            print(f"About to reset; the robot will stand [Press R2 to calibrate]")
            while wait:
                self.button_states = self.command_profile.get_buttons()
                if (
                    self.command_profile.state_estimator.right_lower_right_switch_pressed
                ):
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = (
                        False
                    )
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
                agent.step_once(torch.from_numpy(cal_action))
                agent.get_obs()
                time.sleep(0.05)

            print("Starting pose calibrated [Press R2 to start controller]")
            self.reset_gait_indices()
            while True:
                self.button_states = self.command_profile.get_buttons()
                if (
                    self.command_profile.state_estimator.right_lower_right_switch_pressed
                ):
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = (
                        False
                    )
                    break

        obs = self.agents["hardware_closed_loop"].get_obs()

        gravity_vec = obs[:, 0:3]
        commands = obs[:, 3:6]
        dof_pos = obs[:, 6:18]
        dof_vel = obs[:, 18:30]
        actions = obs[:, 30:42]  # actions are not used in real robot
        # print("=== actions: ", actions)
        contact_states = obs[:, 42:46]

        # print("=== gravity_vec: ", gravity_vec)
        # print("=== contact_states: ", contact_states)

        lin_vel_scale = self.lin_vel_scale
        ang_vel_scale = self.ang_vel_scale
        dof_pos_scale = self.dof_pos_scale
        dof_vel_scale = self.dof_vel_scale

        projected_gravity = gravity_vec
        dof_pos_scaled = dof_pos * dof_pos_scale
        dof_vel_scaled = dof_vel * dof_vel_scale
        commands_scaled = commands * torch.tensor(
            [lin_vel_scale, lin_vel_scale, ang_vel_scale],
            requires_grad=False,
            device=commands.device,
        )

        obs = torch.cat(
            (
                # base_lin_vel,
                # base_ang_vel,
                projected_gravity,
                commands_scaled,
                dof_pos_scaled,
                dof_vel_scaled,
                actions,
                self.clock_inputs
                # self.actions,
                # contact_states
            ),
            dim=-1,
        )

        self.obs_buf[:] = obs

        if self.obs_history:
            self.history_buffer[:] = torch.cat(
                (obs, self.history_buffer[:, self.num_obs :]), dim=1
            )

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        self.lag_buffer = self.lag_buffer[1:] + [self.actions.clone()]
        self.targets = self.lag_buffer[0]

        self.agents["hardware_closed_loop"].publish_action(
            self.targets, hard_reset=False
        )  # default pos will be added inside the agent

        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.control_steps % 100 == 0:
            print(f"frq: {1 / (time.time() - self.time)} Hz", end=" ")
            print(f"control_steps: {self.control_steps}")
        self.time = time.time()

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

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        self.control_steps += 1

        self.obs_dict["obs"] = {
            "state_obs": torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(
                self.rl_device
            )
        }

        if self.obs_history:
            self.obs_dict["obs"]["state_history"] = torch.clamp(
                self.history_buffer, -self.clip_obs, self.clip_obs
            ).to(self.rl_device)

        return (
            self.obs_dict,
            self.rew_buf.to(self.rl_device),
            self.reset_buf.to(self.rl_device),
            self.extras,
        )

    def post_physics_step(self):
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * 3.0, 1.0)

        foot_indices = [
            self.gait_indices + self.phases + self.offsets + self.bounds,
            self.gait_indices + self.offsets,
            self.gait_indices + self.bounds,
            self.gait_indices + self.phases,
        ]

        self.foot_indices = torch.remainder(
            torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0
        )

        # print("clock", self.clock_inputs)
        # print("foots", self.foot_indices)

        # have bug???
        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * self.foot_indices[:, 0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * self.foot_indices[:, 1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * self.foot_indices[:, 2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * self.foot_indices[:, 3])

        self.compute_observations()
        self.compute_reward(self.actions)

        rpy = self.agents[self.control_agent_name].se.get_rpy()
        if abs(rpy[0]) > 1.6 or abs(rpy[1]) > 1.6:
            self.calibrate(wait=False, low=True)

        if self.command_profile.state_estimator.right_lower_right_switch_pressed:
            self.calibrate(wait=False)
            time.sleep(1)
            self.command_profile.state_estimator.right_lower_right_switch_pressed = (
                False
            )
            # self.button_states = self.command_profile.get_buttons()
            while (
                not self.command_profile.state_estimator.right_lower_right_switch_pressed
            ):
                time.sleep(0.01)
                # self.button_states = self.command_profile.get_buttons()
            self.command_profile.state_estimator.right_lower_right_switch_pressed = (
                False
            )

    def compute_reward(self, actions):
        self.rew_buf[0] = 0

    def compute_observations(self):
        obs = self.agents["hardware_closed_loop"].get_obs()

        gravity_vec = obs[:, 0:3]
        commands = obs[:, 3:6]

        self.commands[:, :3] = commands[:, :3]
        # print("command:", commands)
        dof_pos = obs[:, 6:18]
        self.dof_pos = dof_pos + self.default_dof_pos
        self.dof_vel = obs[:, 18:30]
        actions = obs[:, 30:42]  # actions are not used in real robot
        # print("=== actions: ", actions)
        contact_states = obs[:, 42:46]

        # print("=== gravity_vec: ", gravity_vec)
        # print("=== contact_states: ", contact_states)

        lin_vel_scale = self.lin_vel_scale
        ang_vel_scale = self.ang_vel_scale
        dof_pos_scale = self.dof_pos_scale
        dof_vel_scale = self.dof_vel_scale

        projected_gravity = gravity_vec
        dof_pos_scaled = dof_pos * dof_pos_scale
        dof_vel_scaled = self.dof_vel * dof_vel_scale
        commands_scaled = commands * torch.tensor(
            [lin_vel_scale, lin_vel_scale, ang_vel_scale],
            requires_grad=False,
            device=commands.device,
        )

        obs = torch.cat(
            (
                # base_lin_vel,
                # base_ang_vel,
                projected_gravity,
                commands_scaled,
                dof_pos_scaled,
                dof_vel_scaled,
                self.actions,
                self.clock_inputs
                # self.actions,
                # contact_states
            ),
            dim=-1,
        )

        self.obs_buf[:] = obs

        if self.obs_history:
            self.history_buffer[:] = torch.cat(
                (obs, self.history_buffer[:, self.num_obs :]), dim=1
            )

    def reset_idx(self, env_ids):
        pass

    def reset_gait_indices(self, env_ids=None):
        self.gait_indices = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )

        self.clock_inputs = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=torch.float
        )
        if self.obs_history:
            assert self.num_obs == 46, "total_obs_dim should be 43"
            zero_obs = torch.zeros(
                self.num_obs, dtype=torch.float32, device=self.device
            )
            zero_obs[2] = -1  # default gravity
            self.history_per_begin = torch.tile(zero_obs, (self.history_length,))
            self.history_buffer = torch.tile(self.history_per_begin, (self.num_envs, 1))

    def reset(self):
        """Is called only once when environment starts to provide the first observations.
        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """
        self.obs_dict["obs"] = {
            "state_obs": torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(
                self.rl_device
            )
        }

        if self.obs_history:
            self.obs_dict["obs"]["state_history"] = torch.clamp(
                self.history_buffer, -self.clip_obs, self.clip_obs
            ).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict
