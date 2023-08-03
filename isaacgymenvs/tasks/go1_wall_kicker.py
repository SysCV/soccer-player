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
import matplotlib.pyplot as plt

from gym import spaces

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgym import gymutil

from isaacgymenvs.tasks.base.vec_task import VecTask

from isaacgymenvs.utils.torch_jit_utils import calc_heading

from typing import Dict, Any, Tuple, List, Set


class Go1WallKicker(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):



        self.totall_episode = 0
        self.success_episode = 0

        self.cfg = cfg
        # cam pic
        self.have_cam_window = self.cfg["env"]["cameraSensorPlt"]
        self.pixel_obs = self.cfg["env"]["pixel_observations"]["enable"]
        self.state_obs = self.cfg["env"]["state_obs"]
        # print("pixel_obs:", self.pixel_obs)
        if self.have_cam_window:
            _, self.ax = plt.subplots()
            plt.axis('off')
        self.add_real_ball = self.cfg["task"]["target_ball"]

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
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
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
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

        # goal params
        self.goal_init_pos = self.cfg["env"]["goalInitState"]["pos"]
        self.goal_rand_pos_range = self.cfg["env"]["goalInitState"]["randomPosRange"]

        # wall params
        self.wall_init_pos = self.cfg["env"]["wallInitState"]["pos"]
        self.wall_rand_pos_range = self.cfg["env"]["wallInitState"]["randomPosRange"]

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        numObservations = 0
        # print(self.cfg["env"]["state_observations"])
        for v in self.cfg["env"]["state_observations"].values():
            numObservations += v
        self.cfg["env"]["numObservations"] = numObservations

        self.cfg["env"]["numActions"] = self.cfg["env"]["actions_num"]

        # here call _creat_ground and _creat_env
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # rewrite obs space
        if self.pixel_obs:
            self.obs_space = spaces.Dict({"state_obs":spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf),
                                     "pixel_obs":spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
                                    })
            
        if self.state_obs:
            self.obs_space = spaces.Dict({"state_obs":spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
                                    })

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        self._prepare_reward_function()
        self.ball_in_goal_now = torch.zeros(self.num_envs, device=rl_device,dtype=torch.bool)

        self.max_onground_length_s = self.cfg["env"]["learn"]["maxInGroundLength_s"]
        self.max_stable_length_s = self.cfg["env"]["learn"]["maxStableLength_s"]
        self.max_stable_length = int(self.max_stable_length_s / self.dt + 0.5)
        self.max_onground_length = int(self.max_onground_length_s / self.dt + 0.5)

        image_width = self.cfg["env"]["pixel_observations"]["width"]
        image_height = self.cfg["env"]["pixel_observations"]["height"]
        # image_num = self.cfg["env"]["pixel_observations"]["history"]
        # image_channels = 3
        # self.history_images = torch.zeros([self.num_envs, image_num, image_height, image_width, image_channels], dtype=torch.uint8, device=self.device)

        self.create_sim_monitor()
        self.create_self_buffers()


        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # print ("Go1WallKicker init done by gymenv!!")

    def create_sim_monitor(self):
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


    def create_self_buffers(self):
        # initialize some data used later on
        # the differce with monitor is that these are not wrapped gym-state tensors
        self.extras = {}

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1] 
        self.commands_z = self.commands.view(self.num_envs, 3)[..., 2]

        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle


        self.is_back = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.rebound_times = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.target_reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.camera_tensor_imgs_buf = torch.zeros([self.num_envs, 224, 224, 3], dtype=torch.uint8, device=self.device, requires_grad=False)

        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.actor_indices_for_reset: List[torch.Tensor] = []
        self.onground_length = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.stable_length = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

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
        # plane_params.rolling_friction = 0.6
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
        # asset_options.density = 0.001
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
        knee_names = [s for s in body_names if ("hip" in s) or ("thigh" in s)]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        calf_names = [s for s in body_names if ("calf" in s)]
        self.penalised_contact_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(a1)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

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

        if self.pixel_obs:
            self.camera_handles = []
            self.camera_tensor_list = []
            self.frame_count = 0

        # box looking mark
        asset_options_box = gymapi.AssetOptions()
        asset_options_box.density = 1.
        asset_options_box.fix_base_link = True
        asset_box = self.gym.create_box(self.sim, 0.01, 0.3, 0.3, asset_options_box)

        # wall asset
        asset_options_wall = gymapi.AssetOptions()
        asset_options_wall.density = 1.
        asset_options_wall.fix_base_link = True
        asset_wall = self.gym.create_box(self.sim, 0.01, 3, 1, asset_options_wall)
    

        # circle booking mark

        # circle_asset_file = "urdf/circle-urdf/urdf/circle-urdf.urdf"
        # cirvle_asset_options = gymapi.AssetOptions()
        # cirvle_asset_options.fix_base_link = True
        # # cirvle_asset_options.flip_visual_attachments = True
        # circle_asset = self.gym.load_asset(self.sim, asset_root, circle_asset_file, cirvle_asset_options)

        goal_asset = asset_box #  circle_asset

        silver_color = gymapi.Vec3(0.5, 0.5, 0.5)
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            a1_handle = self.gym.create_actor(env_ptr, a1, start_pose, "a1", i, 0b010, 0)

            self.gym.set_actor_dof_properties(env_ptr, a1_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, a1_handle)
            self.envs.append(env_ptr)
            self.a1_handles.append(a1_handle)

            if self.pixel_obs:
                color = gymapi.Vec3(1, 0, 0)
                color_goal = gymapi.Vec3(0, 1, 0)
            else:
                c = 0.7 * np.random.random(3)
                color = gymapi.Vec3(c[0], c[1], c[2])
                color_goal = gymapi.Vec3(c[0], c[1], c[2])


            
            if self.add_real_ball:
                ball_handle = self.gym.create_actor(env_ptr, asset_ball, gymapi.Transform(gymapi.Vec3(*self.ball_init_pos)), "ball", i, 0b001, 1)

                this_ball_props = self.gym.get_actor_rigid_shape_properties(env_ptr,ball_handle)
                this_ball_props[0].rolling_friction = 0.1
                this_ball_props[0].restitution = 0.9
                self.gym.set_actor_rigid_shape_properties(env_ptr, ball_handle, this_ball_props)

                this_ball_phy_props = self.gym.get_actor_rigid_body_properties(env_ptr, ball_handle)
                this_ball_phy_props[0].mass = self.ball_mass
                self.gym.set_actor_rigid_body_properties(env_ptr, ball_handle, this_ball_phy_props)


                self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                self.ball_handles.append(ball_handle)

                # set boxed marker for each env
                goal_handle = self.gym.create_actor(env_ptr, goal_asset, gymapi.Transform(gymapi.Vec3(*self.goal_init_pos)), "box", i, 0b111, 1) # can be asset box

                self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color_goal)
                self.goal_handles.append(goal_handle)

                # set wall for each env
                wall_handle = self.gym.create_actor(env_ptr, asset_wall, gymapi.Transform(gymapi.Vec3(*self.wall_init_pos)), "wall", i, 0b100, 1) # can be asset box
                this_wall_props = self.gym.get_actor_rigid_shape_properties(env_ptr,wall_handle)
                this_wall_props[0].rolling_friction = 0.1
                this_wall_props[0].restitution = 0.99
                self.gym.set_actor_rigid_shape_properties(env_ptr, wall_handle, this_wall_props)

                self.gym.set_rigid_body_color(env_ptr, wall_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.9,0.9,0.9))
                self.wall_handles.append(wall_handle)

            # gymutil.draw_lines(sphere_geom, self.gym, self.viewer, env_ptr, gymapi.Transform(0,1,2))


            # set color for each go1
            self.gym.reset_actor_materials(env_ptr, a1_handle, gymapi.MESH_VISUAL_AND_COLLISION)
            self.gym.set_rigid_body_color(env_ptr, a1_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # set silver for every other part of go1
            for j in range(1,self.gym.get_actor_rigid_body_count(env_ptr, a1_handle)):
                self.gym.set_rigid_body_color(env_ptr, a1_handle, j, gymapi.MESH_VISUAL_AND_COLLISION, color)


            if self.pixel_obs:
                camera_properties = gymapi.CameraProperties()
                camera_properties.horizontal_fov = 130.0
                camera_properties.enable_tensors = True
                camera_properties.width = 224
                camera_properties.height = 224

                cam_handle = self.gym.create_camera_sensor(env_ptr, camera_properties)
                camera_offset = gymapi.Vec3(0.3, 0, 0)
                camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(20))
                body_handle = self.gym.get_actor_rigid_body_handle(env_ptr, a1_handle, 0)

                self.gym.attach_camera_to_body(cam_handle, env_ptr, body_handle, gymapi.Transform(camera_offset, camera_rotation),
                                          gymapi.FOLLOW_TRANSFORM)
                self.camera_handles.append(cam_handle)
                cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)

                # wrap camera tensor in a pytorch tensor
                torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
                self.camera_tensor_list.append(torch_cam_tensor)


        # for i in range(len(feet_names)):
        #     self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], knee_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], "base")

        self.ball_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ball_handles[0], "ball")

        self.goal_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.goal_handles[0], "goal")

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.action_scale * self.actions + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))


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

        if self.pixel_obs:
            self.obs_dict["obs"] = {
                "state_obs":
                torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device),
                "pixel_obs":
                self.camera_tensor_imgs_buf.to(self.rl_device)
            }
        elif self.state_obs:
            self.obs_dict["obs"] = {
            "state_obs":
            torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
            }
        else:
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # for i in self.obs_dict["obs"].values():
        #     print("i th env obs:", i.shape, i.dtype)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        # self.obs_dict["pixel"] = self.history_images.to(self.rl_device)
        # for i in self.obs_dict["obs"].keys():
        #     print("i th obs:", i)

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras
    

    def post_physics_step(self):
        self.progress_buf += 1

        # the reset is from the previous step
        # because we need the observation of 0 step, if compute_reset -> reset, we will lose the observation of 0 step

        goal_env_ids = self.target_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        #print(goal_env_ids)
        if len(goal_env_ids) > 0:
            self.reset_only_target(goal_env_ids)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1) # env_ides is [id1 id2 ...]
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.set_actor_root_state_tensor_indexed()

        self.refresh_self_buffers()




        self.compute_observations()
        self.compute_reset()
        self.compute_reward(self.actions)

        if self.pixel_obs:
            self.compute_pixels()

    def compute_pixels(self):
        self.frame_count += 1
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True) # True means wait for GPU physics to finish, because we need to access the camera results

        self.gym.step_graphics(self.sim)
            # self.render()
            # self.gym.draw_viewer(self.viewer, self.sim, True)

            # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)

        self.gym.start_access_image_tensors(self.sim)

        # self.camera_tensor_imgs = torch.stack(self.camera_tensor_list,dim=0)
        
            # self.history_images = torch.cat([camera_tensor_imgs[:,:,:,:3].unsqueeze(1).clone(),self.history_images[:,0:3,:,:,:3]],dim=1)
        self.camera_tensor_imgs_buf = torch.stack(self.camera_tensor_list,dim=0)[:,:,:,:3].clone()

        self.gym.end_access_image_tensors(self.sim)



        if self.have_cam_window:
            if np.mod(self.frame_count, 1) == 0:
                    #for i in range(len(self.camera_handles)):
                # up_row = torch.cat([self.history_images[0,0,:,:,:],self.history_images[0,1,:,:,:]],dim=1)
                # low_row = torch.cat([self.history_images[0,2,:,:,:],self.history_images[0,3,:,:,:]],dim=1)
                # whole_picture = torch.cat((up_row, low_row), dim=0)
                # cam_img = whole_picture.cpu().numpy()
                cam_img = self.camera_tensor_imgs_buf[0,:,:,:].cpu().numpy()
                self.ax.imshow(cam_img)
                plt.draw()
                plt.pause(0.001)
                self.ax.cla()

    def refresh_self_buffers(self):
        # refresh
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # raw data
        

        # state data
        self.base_quat = self.root_states[::4, 3:7]
        self.base_pos = self.root_states[::4, 0:3]
        self.ball_pos = self.root_states[1::4, 0:3]
        self.goal_pos = self.root_states[2::4, 0:3]
        self.ball_lin_vel_xy_world = self.root_states[1::4, 7:9]

        #judgement data
        is_onground = torch.norm(self.contact_forces[:, self.ball_index, :], dim=1) > 0.1
        self.onground_length[is_onground] += 1
        is_stable = ~(torch.any(self.root_states[1::4, 7:9]> 0.2, dim=1))
        self.stable_length[is_stable] += 1

        self.ball_in_goal_now = (torch.sum(torch.square(self.ball_pos[:,:] - self.goal_pos[:,:]), dim=1) < 0.09)

        self.ball_near_wall_now = torch.abs(self.ball_pos[:,0] - self.wall_init_pos[0]) < 0.2

        self.ball_near_robot_now = (torch.sum(torch.square(self.ball_pos - self.base_pos), dim=1) < 0.5)

        # print("now back:",self.is_back.item(),
        #       "=== now near wall:",self.ball_near_wall_now.item(),
        #         "=== now near robot:",self.ball_near_robot_now.item())


    def compute_reward(self, actions):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        extra_info = {}
        episode_cumulative = {}
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            episode_cumulative[name] = rew
        
        extra_info["episode_cumulative"] = episode_cumulative
        self.extras.update(extra_info)


    def compute_reset(self):
        reset = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1., dim=1)

        reset = reset | (self.ball_pos[:, 0] - self.goal_pos[:, 0] > .1)
        reset = reset | (self.base_pos[:, 0] < self.robot_x_range[0]) | (self.base_pos[:, 0] > self.robot_x_range[1])
        reset = reset | (self.ball_pos[:, 1] < self.ball_y_range[0]) | (self.ball_pos[:, 1] > self.ball_y_range[1])
        reset = reset | (self.ball_pos[:, 0] < self.ball_x_range[0]) | (self.ball_pos[:, 0] > self.ball_x_range[1])

        # reset = reset | self.ball_in_goal_now
        time_out = self.progress_buf >= self.max_episode_length - 1
        onground_time_out = self.onground_length >= self.max_onground_length - 1
        stable_time_out = self.stable_length >= self.max_stable_length - 1
        reset = reset | time_out | onground_time_out | stable_time_out
        
        self.reset_buf[:] = reset

        self.target_reset_buf = self.ball_in_goal_now & ~ self.is_back & ~ self.ball_near_wall_now


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

        base_quat = root_states[::4, 3:7]
        base_pose = root_states[::4, 0:3]
        base_lin_vel = quat_rotate_inverse(base_quat, root_states[::4, 7:10]) * lin_vel_scale
        base_ang_vel = quat_rotate_inverse(base_quat, root_states[::4, 10:13]) * ang_vel_scale
        projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)
        dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

        # commands_scaled = root_states[2::4, 0:3] * torch.tensor(
        #     [lin_vel_scale] * 3,
        #     requires_grad=False,
        #     device=commands.device
        # )
        goal_p = root_states[2::4, 0:3] - root_states[0::4, 0:3]
        ball_states_p = root_states[1::4, 0:3] - root_states[0::4, 0:3]
        ball_states_v = root_states[1::4, 7:10]

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

        if "ball_states_p" in self.cfg["env"]["state_observations"]:
            cat_list.append(ball_states_p)
        if "ball_states_v" in self.cfg["env"]["state_observations"]:
            cat_list.append(ball_states_v)

        if "goal_pose" in self.cfg["env"]["state_observations"]:
            cat_list.append(goal_p)
        if "base_pose" in self.cfg["env"]["state_observations"]:
            cat_list.append(base_pose)
        if "base_quat" in self.cfg["env"]["state_observations"]:
            cat_list.append(base_quat)


        # print(cat_list)
        
        obs = torch.cat(cat_list, dim=-1)

        # obs = torch.cat((
        #     base_lin_vel,
        #     base_ang_vel,
        #     projected_gravity,
        #     commands_scaled,
        #     dof_pos_scaled,
        #     dof_vel * dof_vel_scale,
        #     actions,
        #     ball_states_p,
        #     ball_states_v,
        #     base_pose_privileged,
        #     base_quat
        # ), dim=-1)

        self.obs_buf[:] = obs

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # reward containers
        from isaacgymenvs.tasks.go1func.wall_rewards import RewardTerms
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
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                      requires_grad=False)
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}
    def reset_only_target(self, env_ids):

        num_ids = len(env_ids)
        goal_indices = (env_ids*4+2)
        # print("goal_indices", goal_indices)
        # jump by 2, because a1,ball,a1,ball
        #print("chosen tensor",ball_goal_states[goal_indices, 0])
        self.root_states[goal_indices, 0] = torch.ones([num_ids],device=self.device) * (self.goal_init_pos[0] - self.goal_rand_pos_range[0] / 2.)  + torch.rand([num_ids],device=self.device) * self.goal_rand_pos_range[0] 
        self.root_states[goal_indices, 1] = torch.ones([num_ids],device=self.device) * (self.goal_init_pos[1] - self.goal_rand_pos_range[1] / 2.)  + torch.rand([num_ids],device=self.device) * self.goal_rand_pos_range[1]
        self.root_states[goal_indices, 2] = torch.ones([num_ids],device=self.device) * (self.goal_init_pos[2] - self.goal_rand_pos_range[2] / 2.)+ torch.rand([num_ids],device=self.device) * self.goal_rand_pos_range[2]
     
        # actor_ids_int32 = goal_indices.to(dtype=torch.int32)


        self.commands_x[env_ids] = self.root_states[env_ids*4 + 2, 0]
        self.commands_y[env_ids] = self.root_states[env_ids*4 + 2, 1]
        self.commands_z[env_ids] = self.root_states[env_ids*4 + 2, 2]

        self.deferred_set_actor_root_state_tensor_indexed(goal_indices)

    def reset(self):
        """Is called only once when environment starts to provide the first observations.
        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """

        # self.compute_observations()
        # self.compute_reset()
        # self.compute_reward(self.actions)

        if self.pixel_obs:
            self.compute_pixels()
            self.obs_dict["obs"] = {
                "state_obs":
                torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device),
                "pixel_obs":
                self.camera_tensor_imgs_buf.to(self.rl_device)
            }
        elif self.state_obs:
            self.obs_dict["obs"] = {
            "state_obs":
            torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
            }
        else:
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

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

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        self.onground_length[env_ids] = 0
        self.stable_length[env_ids] = 0
        self.is_back[env_ids] = False





        actor_indices = torch.cat([env_ids*4,env_ids*4+1,env_ids*4+2,env_ids*4+3])

        # print("whole reset index", actor_indices)

        # copy initial robot and ball states
        self.root_states[env_ids*4] = self.initial_root_states[env_ids*4].clone()
        self.root_states[env_ids*4+1] = self.initial_root_states[env_ids*4+1].clone()

        # jump by 2, because a1,ball,a1,ball
        self.root_states[env_ids*4+1, 0] = torch.ones([num_ids],device=self.device) * (self.ball_init_pos[0] - self.ball_rand_pos_range[0] / 2.)  + torch.rand([num_ids],device=self.device) * self.ball_rand_pos_range[0] 
        self.root_states[env_ids*4+1, 1] = torch.ones([num_ids],device=self.device) * (self.ball_init_pos[1] - self.ball_rand_pos_range[1] / 2.)  + torch.rand([num_ids],device=self.device) * self.ball_rand_pos_range[1]
        self.root_states[env_ids*4+1, 2] = torch.ones([num_ids],device=self.device) * (self.ball_init_pos[2] - self.ball_rand_pos_range[2] / 2.) + torch.rand([num_ids],device=self.device) * self.ball_rand_pos_range[2]

        # ball_states[1::3, 7] = torch.ones(self.num_envs) * - 1. * self.ball_init_speed

        # goal_states = self.initial_root_states.clone()
        self.root_states[env_ids*4+2, 0] = torch.ones([num_ids],device=self.device) * (self.goal_init_pos[0] - self.goal_rand_pos_range[0] / 2.)  + torch.rand([num_ids],device=self.device) * self.goal_rand_pos_range[0] 
        self.root_states[env_ids*4+2, 1] = torch.ones([num_ids],device=self.device) * (self.goal_init_pos[1] - self.goal_rand_pos_range[1] / 2.)  + torch.rand([num_ids],device=self.device) * self.goal_rand_pos_range[1]
        self.root_states[env_ids*4+2, 2] = torch.ones([num_ids],device=self.device) * (self.goal_init_pos[2] - self.goal_rand_pos_range[2] / 2.) + torch.rand([num_ids],device=self.device) * self.goal_rand_pos_range[2]

        self.root_states[env_ids*4+3, 0] = torch.ones([num_ids],device=self.device) * (self.wall_init_pos[0] - self.wall_rand_pos_range[0] / 2.)  + torch.rand([num_ids],device=self.device) * self.wall_rand_pos_range[0]
        self.root_states[env_ids*4+3, 1] = torch.ones([num_ids],device=self.device) * (self.wall_init_pos[1] - self.wall_rand_pos_range[1] / 2.)  + torch.rand([num_ids],device=self.device) * self.wall_rand_pos_range[1]
        self.root_states[env_ids*4+3, 2] = torch.ones([num_ids],device=self.device) * (self.wall_init_pos[2] - self.wall_rand_pos_range[2] / 2.)+ torch.rand([num_ids],device=self.device) * self.wall_rand_pos_range[2]


        
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.commands_x[env_ids] = self.root_states[env_ids*4 + 2, 0]
        self.commands_y[env_ids] = self.root_states[env_ids*4 + 2, 1]
        self.commands_z[env_ids] = self.root_states[env_ids*4 + 2, 2]
        # self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        self.deferred_set_actor_root_state_tensor_indexed(actor_indices)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                               gymtorch.unwrap_tensor(self.dof_state),
                                               gymtorch.unwrap_tensor(env_ids_int32 * 4), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

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

