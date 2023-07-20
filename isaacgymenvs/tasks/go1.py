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

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.curricula.curriculum_torch import RewardThresholdCurriculum

from typing import Tuple, Dict

import sys
import time
import math




currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir + '/unitreelib')
import robot_interface as sdk


class Go1(VecTask):
    def control_real_12dof(self, action: torch.Tensor, Kp: float, Kd: float):
        q_target = action[0].clone().cpu().numpy()
        d = self.d
        self.cmd.motorCmd[d['FR_0']].q = q_target[3]
        self.cmd.motorCmd[d['FR_0']].dq = 0
        self.cmd.motorCmd[d['FR_0']].Kp = Kp
        self.cmd.motorCmd[d['FR_0']].Kd = Kd
        self.cmd.motorCmd[d['FR_0']].tau = -0.65

        self.cmd.motorCmd[d['FR_1']].q = q_target[4]
        self.cmd.motorCmd[d['FR_1']].dq = 0
        self.cmd.motorCmd[d['FR_1']].Kp = Kp
        self.cmd.motorCmd[d['FR_1']].Kd = Kd
        self.cmd.motorCmd[d['FR_1']].tau = 0.0

        self.cmd.motorCmd[d['FR_2']].q =  q_target[5]
        self.cmd.motorCmd[d['FR_2']].dq = 0
        self.cmd.motorCmd[d['FR_2']].Kp = Kp
        self.cmd.motorCmd[d['FR_2']].Kd = Kd
        self.cmd.motorCmd[d['FR_2']].tau = 0.0

        self.cmd.motorCmd[d['FL_0']].q = q_target[0]
        self.cmd.motorCmd[d['FL_0']].dq = 0
        self.cmd.motorCmd[d['FL_0']].Kp = Kp
        self.cmd.motorCmd[d['FL_0']].Kd = Kd
        self.cmd.motorCmd[d['FL_0']].tau = -0.65

        self.cmd.motorCmd[d['FL_1']].q = q_target[1]
        self.cmd.motorCmd[d['FL_1']].dq = 0
        self.cmd.motorCmd[d['FL_1']].Kp = Kp
        self.cmd.motorCmd[d['FL_1']].Kd = Kd
        self.cmd.motorCmd[d['FL_1']].tau = 0.0

        self.cmd.motorCmd[d['FL_2']].q =  q_target[2]
        self.cmd.motorCmd[d['FL_2']].dq = 0
        self.cmd.motorCmd[d['FL_2']].Kp = Kp
        self.cmd.motorCmd[d['FL_2']].Kd = Kd
        self.cmd.motorCmd[d['FL_2']].tau = 0.0

        self.cmd.motorCmd[d['RR_0']].q = q_target[9]
        self.cmd.motorCmd[d['RR_0']].dq = 0
        self.cmd.motorCmd[d['RR_0']].Kp = Kp
        self.cmd.motorCmd[d['RR_0']].Kd = Kd
        self.cmd.motorCmd[d['RR_0']].tau = -0.65

        self.cmd.motorCmd[d['RR_1']].q = q_target[10]
        self.cmd.motorCmd[d['RR_1']].dq = 0
        self.cmd.motorCmd[d['RR_1']].Kp = Kp
        self.cmd.motorCmd[d['RR_1']].Kd = Kd
        self.cmd.motorCmd[d['RR_1']].tau = 0.0

        self.cmd.motorCmd[d['RR_2']].q =  q_target[11]
        self.cmd.motorCmd[d['RR_2']].dq = 0
        self.cmd.motorCmd[d['RR_2']].Kp = Kp
        self.cmd.motorCmd[d['RR_2']].Kd = Kd
        self.cmd.motorCmd[d['RR_2']].tau = 0.0

        self.cmd.motorCmd[d['RL_0']].q = q_target[6]
        self.cmd.motorCmd[d['RL_0']].dq = 0
        self.cmd.motorCmd[d['RL_0']].Kp = Kp
        self.cmd.motorCmd[d['RL_0']].Kd = Kd
        self.cmd.motorCmd[d['RL_0']].tau = -0.65

        self.cmd.motorCmd[d['RL_1']].q = q_target[7]
        self.cmd.motorCmd[d['RL_1']].dq = 0
        self.cmd.motorCmd[d['RL_1']].Kp = Kp
        self.cmd.motorCmd[d['RL_1']].Kd = Kd
        self.cmd.motorCmd[d['RL_1']].tau = 0.0

        self.cmd.motorCmd[d['RL_2']].q =  q_target[8]
        self.cmd.motorCmd[d['RL_2']].dq = 0
        self.cmd.motorCmd[d['RL_2']].Kp = Kp
        self.cmd.motorCmd[d['RL_2']].Kd = Kd
        self.cmd.motorCmd[d['RL_2']].tau = 0.0








        self.safe.PowerProtect(self.cmd, self.state, 2)

        self.udp.SetSend(self.cmd)
        self.udp.Send()



    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        # cam pic
        self.save_cam = self.cfg["task"]["save_cam_pic"]
        self.add_fake_ball = self.cfg["task"]["fake_ball"]
        self.real = self.cfg["env"]["real_robot"]

        if self.real:
            self.d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
                'FL_0':3, 'FL_1':4, 'FL_2':5, 
                'RR_0':6, 'RR_1':7, 'RR_2':8, 
                'RL_0':9, 'RL_1':10, 'RL_2':11 }
            LOWLEVEL  = 0xff


            self.udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
            self.safe = sdk.Safety(sdk.LeggedType.Go1)
            self.cmd = sdk.LowCmd()
            self.state = sdk.LowState()
            self.safe.PowerProtect(self.cmd, self.state, 2)
            self.udp.InitCmdData(self.cmd)

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

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

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

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

        # init curriculum
        self.curriculum = RewardThresholdCurriculum(seed=42, device=self.device,
                x_vel=(self.cfg["env"]["randomCommandVelocityRanges"]           ["linear_x"][0],
                        self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"][1],
                        self.cfg["env"]["randomCommandVelocityRanges"]["num_bins_x"]),
                y_vel=(self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"][0],
                        self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"][1],
                        self.cfg["env"]["randomCommandVelocityRanges"]["num_bins_y"]),
                yaw_vel=(self.cfg["env"]["randomCommandVelocityRanges"]["yaw"][0],
                        self.cfg["env"]["randomCommandVelocityRanges"]["yaw"][1],
                        self.cfg["env"]["randomCommandVelocityRanges"]["num_bins_yaw"]))
        

        low = torch.tensor([-0.5, -0.2, -0.1], device=self.device)
        high = torch.tensor([0.5, 0.2, 0.1], device=self.device)

        self.curriculum.set_to(low,high)

        # self.env_command_bins = np.zeros(self.num_envs, dtype=np.int)
        _, self.env_command_bins = self.curriculum.sample_bins(self.num_envs)

        # print("env_command_bins is init as: ", self.env_command_bins)
        self.task_rewards_episode = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.task_rewards_threshold = 5.0
        self.have_plt_curriculum = False
        self.total_train_step = 0




        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/go1/urdf/go1.urdf"
        #asset_path = os.path.join(asset_root, asset_file)
        #asset_root = os.path.dirname(asset_path)
        #asset_file = os.path.basename(asset_path)

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
        # extremity_name = "SHANK" if asset_options.collapse_fixed_joints else "FOOT"
        # extremity_name = "foot"
        # feet_names = [s for s in body_names if extremity_name in s]
        # self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "thigh" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(a1)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_options = gymapi.AssetOptions()
        asset_options.density = 1.
        asset_ball = self.gym.create_sphere(self.sim, 0.1, asset_options)

        self.a1_handles = []
        self.envs = []
        if self.add_fake_ball:
            self.ball_handles = []

        if self.save_cam:
            self.camera_handles = []
            if not os.path.exists("a1_images"):
                os.mkdir("a1_images")
            self.frame_count = 0

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            a1_handle = self.gym.create_actor(env_ptr, a1, start_pose, "a1", i, 0, 0)

            self.gym.set_actor_dof_properties(env_ptr, a1_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, a1_handle)
            self.envs.append(env_ptr)
            self.a1_handles.append(a1_handle)

            c = 0.5 * np.random.random(3)
            color = gymapi.Vec3(c[0], c[1], c[2])

            if self.add_fake_ball:
                ball_handle = self.gym.create_actor(env_ptr, asset_ball, start_pose, "ball", i, 0, 1)
                self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                self.ball_handles.append(ball_handle)

            self.gym.set_rigid_body_color(env_ptr, a1_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)



            if self.save_cam:
                camera_properties = gymapi.CameraProperties()
                camera_properties.width = 360
                camera_properties.height = 240

                h2 = self.gym.create_camera_sensor(env_ptr, camera_properties)
                camera_offset = gymapi.Vec3(1, 0, 0.5)
                camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(0))
                body_handle = self.gym.get_actor_rigid_body_handle(env_ptr, a1_handle, 0)

                self.gym.attach_camera_to_body(h2, env_ptr, body_handle, gymapi.Transform(camera_offset, camera_rotation),
                                          gymapi.FOLLOW_TRANSFORM)
                self.camera_handles.append(h2)


        # for i in range(len(feet_names)):
        #     self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], knee_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "base")

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.action_scale * self.actions + self.default_dof_pos

        if self.real:
            print("==== this time DOF targets",targets[0,:])
            time.sleep(0.005)
            self.control_real_12dof(targets, 15., 1)

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1) # env_ides is [id1 id2 ...]
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.save_cam:
            self.frame_count += 1

        if self.save_cam:
            self.gym.step_graphics(self.sim)

            # render the camera sensors
            self.gym.render_all_camera_sensors(self.sim)
            if np.mod(self.frame_count, 1) == 0:
                i = 2
                rgb_filename = "a1_images/rgb_env%d_frame%d.png" % (i, self.frame_count)
                self.gym.write_camera_image_to_file(
                    self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR, rgb_filename)

    def compute_reward(self, actions):
        self.total_train_step += 1
        # self.rew_buf[:], self.reset_buf[:] = compute_anymal_reward(
        self.rew_buf[:], self.reset_buf[:], extra_info_to_log = compute_anymal_reward(
                #
            # tensors
            self.root_states[:,:],
            # if have balls added, this should be [::2,:],
            # because a1 ball a1 ball ...
            self.commands,
            self.torques,
            self.contact_forces,
            self.knee_indices,
            self.progress_buf,
            # Dict
            self.rew_scales,
            # other
            self.base_index,
            self.max_episode_length,
        )
        self.task_rewards_episode += self.rew_buf
        self.extras.update(extra_info_to_log)
        # self.extras["double"] = extra_info_to_log

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

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

        base_quat = root_states[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
        base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
        projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)
        dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

        commands_scaled = commands*torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale], requires_grad=False, device=commands.device)


        obs = torch.cat((
            #base_lin_vel,
            #base_ang_vel,
            projected_gravity,
            commands_scaled,
            dof_pos_scaled,
            dof_vel * dof_vel_scale,
            actions,
        ), dim=-1)

        self.obs_buf[:] = obs


    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        batch_size = len(env_ids)
        
        if batch_size != 0: 

            old_bins = self.env_command_bins[env_ids]
            # print("old_bins_is",old_bins)

            self.curriculum.update(old_bins, [self.task_rewards_episode[env_ids]], [self.task_rewards_threshold],
                                  local_range=torch.tensor(
                                      [0.6,0.3,0.3], device=self.device))
            
            if self.total_train_step % (50 * 24) == 0 and not self.have_plt_curriculum:
                self.have_plt_curriculum = True
                print("plotting curriculum size", self.curriculum.weights_shaped.shape)

                if wandb.run is not None:
                    wandb.log({"curriculum": plt.imshow(torch.sum(self.curriculum.weights_shaped, axis=2).cpu(), cmap='gray').get_figure()})
                # self.extras.update({"curriculum": plt.imshow(np.sum(self.curriculum.weights_shaped, axis=2), cmap='gray').get_figure()})
            else:
                self.have_plt_curriculum = False
                # self.extras.update({"curriculum": None})
            

            new_commands, new_bin_inds = self.curriculum.sample(batch_size=batch_size)
            self.env_command_bins[env_ids] = new_bin_inds
            self.commands[env_ids, :] = torch.tensor(new_commands[:, :]).to(self.device)



        # self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        # self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        # self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        if self.add_fake_ball:
            ball_states = self.initial_root_states.clone()
            ball_states[1::2, 0] = self.commands_x * 4
            ball_states[1::2, 1] = self.commands_y * 4

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.initial_root_states),
                                                         gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


        # TODO: need debug here!!!
        # dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        #
        # print("Gym dof tensor shape:", dof_tensor.shape)
        # print("Gym dof tensor data @ 0x%x" % dof_tensor.data_address)
        #
        # # wrap physics state tensor in a pytorch tensor
        # dof_torch_tensor = gymtorch.wrap_tensor(dof_tensor)
        # print("Torch state tensor device:", dof_torch_tensor.device)
        # print("Torch state tensor shape:", dof_torch_tensor.shape)
        # print("Torch state tensor data @ 0x%x" % dof_torch_tensor.data_ptr())
        #
        # print("self state tensor shape:", self.dof_state.shape)

        # self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                               gymtorch.unwrap_tensor(self.dof_state),
                                               gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # print("shape",(len(env_ids), 1))
        # print("self device now ::", self.device)
        # print(torch.rand(1024,1, device=self.device))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.task_rewards_episode[env_ids] = 0


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_anymal_reward(
    # tensors
    root_states,
    commands,
    torques,
    contact_forces,
    knee_indices,
    episode_lengths,
    # Dict
    rew_scales,
    # other
    base_index,
    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int) -> Tuple[Tensor, Tensor, Dict[str, float]]

    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # prepare quantities (TODO: return from obs ?)
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])
    extra_info:Dict[str, float] = {} # Dict[str, float]

    # velocity tracking reward
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]
    rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * rew_scales["ang_vel_z"]
    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]

    total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_torque
    # print("shape!!",rew_lin_vel_xy.shape)
    # print("item!!",torch.sum(rew_lin_vel_xy).item())
    extra_info["rew_lin_vel_xy"] = torch.sum(rew_lin_vel_xy).item()
    extra_info["rew_ang_vel_z_sum"] = torch.sum(rew_ang_vel_z).item()
    extra_info["rew_torque"] = torch.sum(rew_torque).item()

    total_reward = torch.clip(total_reward, 0., None)
    # reset agents
    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    reset = reset | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)
    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    reset = reset | time_out

    return total_reward.detach(), reset, extra_info


