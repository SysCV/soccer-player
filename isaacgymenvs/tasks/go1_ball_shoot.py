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

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgym import gymutil

from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import calc_heading

from typing import Dict, Any, Tuple, List, Set


class Go1BallShoot(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.totall_episode = 0
        self.success_episode = 0

        self.cfg = cfg
        # cam pic
        self.save_cam = self.cfg["task"]["save_cam_pic"]
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

        # TODO: out-dated!!!
        # self.rew_scales = {}
        # self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        # self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        # self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]

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

        # ball params
        self.base_init_state = state
        self.ball_init_pos = self.cfg["env"]["ballInitState"]["pos"]
        self.ball_init_speed = self.cfg["env"]["ballInitState"]["ballInitSpeed"]

        # goal params
        self.goal_init_z = self.cfg["env"]["goalInitState"]["height"]

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 51 # todo: change here?
        # 3 base_v + 3 base_w + 3 g + 2 (command_goal_on_ground) + 12(dof_p) + 12(dof_v) + 12 act + 4 (ball pose and velocity)
        self.cfg["env"]["numActions"] = 12 # todo:change here?

        # here call _creat_ground and _creat_env
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        self.max_onground_length_s = self.cfg["env"]["learn"]["maxInGroundLength_s"]
        self.max_stable_length_s = self.cfg["env"]["learn"]["maxStableLength_s"]
        self.max_stable_length = int(self.max_stable_length_s / self.dt + 0.5)
        self.max_onground_length = int(self.max_onground_length_s / self.dt + 0.5)

        self.onground_length = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.stable_length = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)



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

        self.commands = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 2)[..., 1] # not clone! so change here = change command
        self.commands_x = self.commands.view(self.num_envs, 2)[..., 0]
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
        knee_names = [s for s in body_names if "hip" in s]
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
        asset_options.density = 0.1
        asset_ball = self.gym.create_sphere(self.sim, 0.1, asset_options)



        

        self.a1_handles = []
        self.envs = []
        if self.add_real_ball:
            self.ball_handles = []
        
        self.goal_handles = []

        if self.save_cam:
            self.camera_handles = []
            if not os.path.exists("a1_images"):
                os.mkdir("a1_images")
            self.frame_count = 0

        # box looking mark
        asset_options_box = gymapi.AssetOptions()
        asset_options_box.density = 1.
        asset_options_box.fix_base_link = True
        asset_box = self.gym.create_box(self.sim, 0.01, 0.3, 0.1, asset_options_box)

        # circle booking mark

        # circle_asset_file = "urdf/circle-urdf/urdf/circle-urdf.urdf"
        # cirvle_asset_options = gymapi.AssetOptions()
        # cirvle_asset_options.fix_base_link = True
        # # cirvle_asset_options.flip_visual_attachments = True
        # circle_asset = self.gym.load_asset(self.sim, asset_root, circle_asset_file, cirvle_asset_options)

        goal_asset = asset_box #  circle_asset


        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            a1_handle = self.gym.create_actor(env_ptr, a1, start_pose, "a1", i, 0b010, 0)

            self.gym.set_actor_dof_properties(env_ptr, a1_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, a1_handle)
            self.envs.append(env_ptr)
            self.a1_handles.append(a1_handle)

            c = 0.5 * np.random.random(3)
            color = gymapi.Vec3(c[0], c[1], c[2])


            
            if self.add_real_ball:
                ball_handle = self.gym.create_actor(env_ptr, asset_ball, start_pose, "ball", i, 0b001, 1)

                this_ball_props = self.gym.get_actor_rigid_shape_properties(env_ptr,ball_handle)
                this_ball_props[0].rolling_friction = 0.1
                this_ball_props[0].restitution = 0.9
                self.gym.set_actor_rigid_shape_properties(env_ptr, ball_handle, this_ball_props)

                # this_ball_phy_props = self.gym.get_actor_rigid_body_properties(env_ptr, ball_handle)
                # this_ball_phy_props[0].mass = 0.45
                # self.gym.set_actor_rigid_body_properties(env_ptr, ball_handle, this_ball_phy_props)


                self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                self.ball_handles.append(ball_handle)

                # set boxed marker for each env

                goal_handle = self.gym.create_actor(env_ptr, goal_asset, gymapi.Transform(gymapi.Vec3(2,0,self.goal_init_z)), "box", i, 0b111, 1) # can be asset box

                self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                self.goal_handles.append(goal_handle)

            # gymutil.draw_lines(sphere_geom, self.gym, self.viewer, env_ptr, gymapi.Transform(0,1,2))


            # set color for each a1
            self.gym.set_rigid_body_color(env_ptr, a1_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)


            if self.save_cam:
                camera_properties = gymapi.CameraProperties()
                camera_properties.enable_tensors = True
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

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def post_physics_step(self):
        self.progress_buf += 1

        # the reset is from the previous step
        # because we need the observation of 0 step, if compute_reset -> reset, we will lose the observation of 0 step
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1) # env_ides is [id1 id2 ...]
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.refresh_self_buffers()

        self.compute_observations()
        self.compute_reset()
        self.compute_reward(self.actions)

        if self.save_cam:
            self.frame_count += 1

        if self.save_cam:
            self.gym.step_graphics(self.sim)

            # render the camera sensors
            self.gym.render_all_camera_sensors(self.sim)

            self.gym.start_access_image_tensors(self.sim)

            # test_x = torch.randn(3, 4, 5, device='cuda:2')
            # print("test device is on [2]?!!!! is -- %d"%test_x.get_device())

            if np.mod(self.frame_count, 1) == 0:
                for i in range(len(self.camera_handles)):
                    print("generate image: " + "a1_images/rgb_env%d_frame%d.png" % (i, self.frame_count))
                    rgb_filename = "a1_images/rgb_env%d_frame%d.png" % (i, self.frame_count)
                    # self.gym.write_camera_image_to_file(
                    #     self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR, rgb_filename)
                    cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR)
                    torch_camera_tensor: torch.Tensor = gymtorch.wrap_tensor(cam_tensor)

                    print(torch_camera_tensor.get_device())
                    # print(self.root_states.get_device())
            self.gym.end_access_image_tensors(self.sim)

    def refresh_self_buffers(self):
        # refresh
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # raw data
        

        # state data
        self.base_quat = self.root_states[::3, 3:7]
        self.base_pos = self.root_states[::3, 0:3]
        self.ball_pos = self.root_states[1::3, 0:3]
        self.goal_pos = self.root_states[2::3, 0:3]
        self.ball_lin_vel_xy_world = self.root_states[1::3, 7:9]

        #judgement data
        is_onground = torch.norm(self.contact_forces[:, self.ball_index, :], dim=1) > 0.1
        self.onground_length[is_onground] += 1
        is_stable = ~(torch.any(self.root_states[1::3, 7:9]> 0.2, dim=1))
        self.stable_length[is_stable] += 1
        self.ball_in_goal_now = (torch.sum(torch.square(self.ball_pos - self.goal_pos), dim=1) < 0.05)


    def compute_reward_prev(self):
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


    def compute_reward(self, actions):

        extra_info = {}

        ball_goal_distance_error = torch.sum(torch.square(self.ball_pos - self.goal_pos), dim=1)


        reward_ball_in_goal = torch.zeros_like(self.ball_in_goal_now, dtype=torch.float, device=self.ball_in_goal_now.device)
        reward_ball_in_goal[self.ball_in_goal_now] = 1000.

        rew_distance = 10 * torch.exp(-ball_goal_distance_error)

        ball_dog_distance_error = torch.sum(torch.square(self.ball_pos - self.base_pos), dim=1)
        ball_dog_distance_error = torch.clamp_min(ball_dog_distance_error, 0.5)
        rew_ball_dog_distance = 2 * torch.exp(-ball_dog_distance_error)

        robot_heading = torch.tensor([[1., 0., 0.]] * self.base_pos.size(0), device=self.root_states.device)
        base_quat_world = quat_rotate(self.base_quat, robot_heading)
        base_to_ball_world = self.ball_pos - self.base_pos
        base_quat_world_xy = base_quat_world[:, :3]
        base_to_ball_world_xy = base_to_ball_world[:, :3]
        base_quat_world_xy = base_quat_world_xy / torch.norm(base_quat_world_xy, dim=1, keepdim=True)
        base_to_ball_world_xy = base_to_ball_world_xy / torch.norm(base_to_ball_world_xy, dim=1, keepdim=True)
        dot_product = torch.sum(base_quat_world_xy * base_to_ball_world_xy, dim=1)
        heading_reward = torch.exp(dot_product) / 200.

        speed_error = torch.sum(torch.square(self.ball_lin_vel_xy_world - self.commands), dim=1)
        rew_ball_speed = torch.exp(-speed_error)

        ball_z = self.ball_pos[:, 2]
        ball_z_reward = torch.exp(ball_z) / 4.


        ball_speed_square = torch.sum(torch.square(self.ball_lin_vel_xy_world), dim=1)
        command_speed_square = torch.sum(torch.square(self.commands), dim=1)
        rew_have_speed = torch.tanh(torch.clamp_max(ball_speed_square, command_speed_square))

        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

        total_reward = heading_reward + rew_distance + rew_have_speed + \
            rew_torque \
        # + ball_z_reward 
        + rew_ball_dog_distance + reward_ball_in_goal

        extra_info["each_reward/rew_distance"] = torch.sum(rew_distance).item()
        extra_info["each_reward/rew_ball_dog_distance"] = torch.sum(rew_ball_dog_distance).item()
        extra_info["each_reward/have_speed"] = torch.sum(rew_have_speed).item()
        extra_info["each_reward/less_torque_reward"] = torch.sum(rew_torque).item()
        extra_info["each_reward/ball_z_reward"] = torch.sum(ball_z_reward).item()

        total_reward = torch.clip(total_reward, 0., None)


        # ball_in_goal_count = torch.sum(self.ball_in_goal_now.int()).item()
        # total_count = torch.sum(reset.int()).item()

        # self.totall_episode += total_count
        # self.success_episode += ball_in_goal_count

        # extra_info["total_episode_this_epoch"] = total_count
        # extra_info["succeed_episode_this_epoch"] = ball_in_goal_count

        self.rew_buf[:] = total_reward.detach()
        self.extras.update(extra_info)

    def compute_reset(self):
        reset = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.
        reset = reset | torch.any(torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1., dim=1)

        reset = reset | (self.ball_pos[:, 0] - self.goal_pos[:, 0] > .1)
        reset = reset | self.ball_in_goal_now

        time_out = self.progress_buf >= self.max_episode_length - 1
        onground_time_out = self.onground_length >= self.max_onground_length - 1
        stable_time_out = self.stable_length >= self.max_stable_length - 1
        reset = reset | time_out | onground_time_out | stable_time_out
        
        self.reset_buf[:] = reset


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

        base_quat = root_states[::3, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, root_states[::3, 7:10]) * lin_vel_scale
        base_ang_vel = quat_rotate_inverse(base_quat, root_states[::3, 10:13]) * ang_vel_scale
        projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)
        dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

        commands_scaled = root_states[2::3, 0:2] * torch.tensor(
            [lin_vel_scale, lin_vel_scale],
            requires_grad=False,
            device=commands.device
        )

        ball_states_p = root_states[1::3, 0:2] - root_states[0::3, 0:2]
        ball_states_v = root_states[1::3, 8:10]

        obs = torch.cat((
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            commands_scaled,
            dof_pos_scaled,
            dof_vel * dof_vel_scale,
            actions,
            ball_states_p,
            ball_states_v
        ), dim=-1)

        self.obs_buf[:] = obs

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # reward containers
        from tasks.go1.shoot_rewards import RewardTerms
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

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        self.onground_length[env_ids] = 0
        self.stable_length[env_ids] = 0





        actor_indices = torch.cat([env_ids*3,env_ids*3+1,env_ids*3+2]).clone()

        if self.add_real_ball:
            ball_states = self.initial_root_states.clone()
            ball_states[1::3, 0] = torch.ones([self.num_envs]) * 0.5 + torch.rand([self.num_envs]) * 0.2 # jump by 2, because a1,ball,a1,ball
            ball_states[1::3, 1] = torch.ones([self.num_envs]) * -0.5 + torch.rand([self.num_envs]) * 1.
            ball_states[1::3, 2] = torch.ones(self.num_envs) * 0.15

            # ball_states[1::3, 7] = torch.ones(self.num_envs) * - 1. * self.ball_init_speed

            # goal_states = self.initial_root_states.clone()
            ball_states[2::3, 0] = torch.ones([self.num_envs]) * 2 + torch.rand([self.num_envs]) * 0.5
            ball_states[2::3, 1] = torch.rand([self.num_envs]) * 2 + torch.ones([self.num_envs]) * (- 1)
            ball_states[2::3, 2] = torch.ones([self.num_envs]) * 0.1
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        actot_ids_int32 = actor_indices.to(dtype=torch.int32)

        self.commands_x[env_ids] = ball_states[env_ids*3 + 2, 0]
        self.commands_y[env_ids] = ball_states[env_ids*3 + 2, 1]
        # self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        # balls
        if self.add_real_ball:

            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(ball_states),
                                                         gymtorch.unwrap_tensor(actot_ids_int32), len(actot_ids_int32))
            
            # self.gym.set_actor_root_state_tensor_indexed(self.sim,
            #         gymtorch.unwrap_tensor(ball_states),
            #         gymtorch.unwrap_tensor(env_ids_int32 * 3 + 2), len(env_ids_int32))

            # self.gym.set_actor_root_state_tensor_indexed(self.sim,
            #                                          gymtorch.unwrap_tensor(ball_states),
            #                                          gymtorch.unwrap_tensor(env_ids_int32*3 + 1), len(env_ids_int32))
            

            


        else:
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.initial_root_states),
                                                         gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


        # need debug here ===> solved, env_ids_int32 should *2

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                               gymtorch.unwrap_tensor(self.dof_state),
                                               gymtorch.unwrap_tensor(env_ids_int32 * 3), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1


#####################################################################
###=========================jit functions=========================###
#####################################################################


def compute_anymal_reward(
    # tensors
    self,
    root_states,
    commands,
    torques,
    contact_forces,
    knee_indices,
    episode_lengths,
    onground_lengths,
    stable_lengths,
    # Dict
    rew_scales,
    # other
    base_index,
    max_episode_length,
    max_onground_length,
    max_stable_length):
    # (reward, reset, feet_in air, feet_air_time, episode sums)

        # prepare quantities (TODO: return from obs ?)
        base_quat = root_states[::3, 3:7]
        base_pos = root_states[::3, 0:3]
        ball_pos = root_states[1::3, 0:3]
        goal_pos = root_states[2::3, 0:3]
        ball_lin_vel_xy_world = root_states[1::3, 7:9] # should multi by quat_inv to be xy_base
        # base_lin_vel_base = quat_rotate_inverse(base_quat, root_states[::2, 7:10]) # should debug here!!!
        # base_ang_vel = quat_rotate_inverse(base_quat, root_states[::2, 10:13])
        extra_info:Dict[str, float] = {} # Dict[str, float]

        # velocity tracking reward
        # lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        # ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
        # rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]
        # rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * rew_scales["ang_vel_z"]

        # get goal close to ball
        # distance_error = torch.sum(torch.square(ball_pos - base_pos), dim=1)
        ball_goal_distance_error = torch.sum(torch.square(ball_pos - goal_pos), dim=1)

        ball_in_goal = (ball_goal_distance_error < 0.1)

        reward_ball_in_goal = torch.zeros_like(ball_in_goal, dtype=torch.float, device=ball_in_goal.device)

        reward_ball_in_goal[ball_in_goal] = 1000.
        # print("ball in the goal!!",count_in)

        rew_distance = 10 * torch.exp(-ball_goal_distance_error)

        # ball dog distance error 
        ball_dog_distance_error = torch.sum(torch.square(ball_pos - base_pos), dim=1)

        # clip the distance error smaller than 0.5
        ball_dog_distance_error = torch.clamp_min(ball_dog_distance_error, 0.5)

        rew_ball_dog_distance = 2 * torch.exp(-ball_dog_distance_error)


        # heading to the ball
        robot_heading = torch.tensor([[1., 0., 0.]] * base_pos.size(0),device=root_states.device)
        base_quat_world = quat_rotate(base_quat, robot_heading)
        base_to_ball_world = ball_pos - base_pos

        base_quat_world_xy = base_quat_world[:, :3]
        base_to_ball_world_xy = base_to_ball_world[:, :3]

        # normalize the vectors
        base_quat_world_xy = base_quat_world_xy / torch.norm(base_quat_world_xy, dim=1, keepdim=True)
        base_to_ball_world_xy = base_to_ball_world_xy / torch.norm(base_to_ball_world_xy, dim=1, keepdim=True)

        # compute the dot product
        dot_product = torch.sum(base_quat_world_xy * base_to_ball_world_xy, dim=1)
        
        heading_reward = torch.exp(dot_product) / 200.


        # ball follow command
        speed_error = torch.sum(torch.square(ball_lin_vel_xy_world - commands), dim=1)
        rew_ball_speed = torch.exp(-speed_error)

        # reward ball in the air
        ball_z = ball_pos[:, 2]

        # print(ball_z.max().item())
        
        ball_z_reward = torch.exp(ball_z) / 4.

        # reward successfully goal
        if_goal = ball_goal_distance_error < 0.2

        # but first need the ball to move
        ball_speed_square = torch.sum(torch.square(ball_lin_vel_xy_world), dim=1)
        command_speed_square = torch.sum(torch.square(commands), dim=1)
        incorage_speed = torch.clamp_max(ball_speed_square, ball_speed_square)
        rew_have_speed = torch.tanh(incorage_speed)

        # head to ball
        # headings = calc_heading(base_quat)
        # print("===== ball in base ========",ball_pos - base_pos)

        # ball_base = quat_rotate_inverse(base_quat, ball_pos - base_pos)

        # rotation_error = calc_heading(ball_base)
        # # print(rotation_error)

        # rew_head_rotation = torch.exp(- torch.square(rotation_error))

        # torque penalty
        rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]

        # total_reward = ball_z_reward
        
        total_reward = heading_reward + rew_distance \
        + rew_have_speed + rew_torque + ball_z_reward \
        + rew_ball_dog_distance + reward_ball_in_goal #+ rew_ball_speed \
        # print("shape!!",rew_lin_vel_xy.shape)
        # print("item!!",torch.sum(rew_lin_vel_xy).item())
        extra_info["each_reward/rew_distance"] = torch.sum(rew_distance).item()
        extra_info["each_reward/rew_ball_dog_distance"] = torch.sum(rew_ball_dog_distance).item()
        # extra_info["each_reward/speed_follow"] = torch.sum(rew_ball_speed).item()
        extra_info["each_reward/have_speed"] = torch.sum(rew_have_speed).item()
        # extra_info["rew_ang_vel_z_sum"] = torch.sum(rew_ang_vel_z).item()
        extra_info["each_reward/less_torque_reward"] = torch.sum(rew_torque).item()
        extra_info["each_reward/ball_z_reward"] = torch.sum(ball_z_reward).item()
        # extra_info["each_reward/heading_reward"] = torch.sum(heading_reward).item()



        total_reward = torch.clip(total_reward, 0., None)
        # reset agents
        reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1. # because each dog only have 1 base
        reset = reset | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1) # because at least 4 knees, so knee_indices is a list

        # reset = reset | (torch.norm(ball_pos - base_pos, dim=1) > 3.) # too far way
        reset = reset | (ball_pos[:,0] - goal_pos[:,0] > .1) # further than the goal

        reset = reset | ball_in_goal # ball in the goal

        time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
        onground_time_out = onground_lengths >= max_onground_length - 1
        stable_time_out = stable_lengths >= max_stable_length - 1
        reset = reset | time_out | onground_time_out | stable_time_out


        ball_in_goal_count = torch.sum(ball_in_goal.int()).item()
        total_count = torch.sum(reset.int()).item()

        self.totall_episode += total_count
        self.success_episode += ball_in_goal_count

        # print("total_count",self.totall_episode)
        # print("ball_in_goal_count",self.success_episode)

        extra_info["total_episode_this_epoch"] = total_count
        extra_info["succeed_episode_this_epoch"] = ball_in_goal_count
        # for testing
        # reset.fill_(0)

        return total_reward.detach(), reset, extra_info


@torch.jit.script
def compute_anymal_observations(root_states,
                                commands,
                                dof_pos,
                                default_dof_pos,
                                dof_vel,
                                gravity_vec,
                                actions,
                                lin_vel_scale,
                                ang_vel_scale,
                                dof_pos_scale,
                                dof_vel_scale
                                ):

    # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float) -> torch.Tensor
    base_quat = root_states[::3, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[::3, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[::3, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    # commands_scaled = commands*torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale], requires_grad=False, device=commands.device)
    commands_scaled = root_states[2::3, 0:2] * torch.tensor([lin_vel_scale, lin_vel_scale], requires_grad=False,
                                              device=commands.device)

    ball_states_p = root_states[1::3, 0:2] - root_states[0::3, 0:2] #  ball_p - a1_p
    ball_states_v = root_states[1::3, 8:10]

    obs = torch.cat((base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     commands_scaled,
                     dof_pos_scaled,
                     dof_vel*dof_vel_scale,
                     actions,
                     ball_states_p,
                     ball_states_v
                     ), dim=-1)

    # 3 base_v + 3 base_w + 3 g + 2 (command) + 12(dof_p) + 12(dof_v) + 12 act + 4 ball

    return obs
