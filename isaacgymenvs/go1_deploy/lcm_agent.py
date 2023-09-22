import time

import lcm
import numpy as np
import torch
import cv2

from isaacgymenvs.go1_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class LCMAgent:
    def __init__(self, cfg, se, command_profile, device="cpu"):
        if not isinstance(cfg, dict):
            cfg = class_to_dict(cfg)
        self.cfg = cfg
        self.se = se
        self.command_profile = command_profile

        self.dt = self.cfg["sim"]["dt"]
        self.timestep = 0

        self.num_obs = 3 + 3 + 12 + 12 + 12 + 4
        self.num_envs = 1
        self.num_privileged_obs = 0
        self.num_actions = self.cfg["env"]["act_num"]
        self.num_commands = 3
        self.device = device

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
        self.default_dof_pos = np.array(
            [self.cfg["env"]["defaultJointAngles"][name] for name in joint_names]
        )
        self.default_dof_pos_scale = np.ones(12)
        self.default_dof_pos = self.default_dof_pos * self.default_dof_pos_scale

        self.p_gains = np.zeros(12)
        self.d_gains = np.zeros(12)
        for i in range(12):
            self.p_gains[i] = 20
            self.d_gains[i] = 0.5
            found = True

        print(f"p_gains: {self.p_gains}")

        self.commands = np.zeros((1, self.num_commands))
        self.actions = torch.zeros(12)
        self.last_actions = torch.zeros(12)
        self.gravity_vector = np.zeros(3)
        self.dof_pos = np.zeros(12)
        self.dof_vel = np.zeros(12)
        self.body_linear_vel = np.zeros(3)
        self.body_angular_vel = np.zeros(3)
        self.joint_pos_target = np.zeros(12)
        self.joint_vel_target = np.zeros(12)
        self.torques = np.zeros(12)
        self.contact_state = np.ones(4)

        self.joint_idxs = self.se.joint_idxs

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float)

        self.is_currently_probing = False

    def set_probing(self, is_currently_probing):
        self.is_currently_probing = is_currently_probing

    def get_obs(self):
        cmds, reset_timer = self.command_profile.get_command(
            self.timestep * self.dt, probe=self.is_currently_probing
        )
        self.commands[:, :] = cmds[: self.num_commands]
        if reset_timer:
            self.reset_gait_indices()

        self.gravity_vector = self.se.get_gravity_vector()
        self.dof_pos = self.se.get_dof_pos()
        self.dof_vel = self.se.get_dof_vel()
        self.body_linear_vel = self.se.get_body_linear_vel()
        self.body_angular_vel = self.se.get_body_angular_vel()
        self.contact_state = self.se.get_contact_state()

        ob = np.concatenate(
            (
                self.gravity_vector.reshape(1, -1),
                self.commands,
                (self.dof_pos - self.default_dof_pos).reshape(1, -1),
                self.dof_vel.reshape(1, -1),
                self.actions.cpu().detach().numpy().reshape(1, -1),
            ),
            axis=1,
        )

        if (
            "observe_contact_states" in self.cfg["env"].keys()
            and self.cfg["env"]["observe_contact_states"]
        ):
            ob = np.concatenate((ob, self.contact_state.reshape(1, -1)), axis=-1)

        return torch.tensor(ob, device=self.device).float()

    def get_privileged_observations(self):
        return self.se.get_rms()

    def publish_action(self, action, hard_reset=False):
        self.actions = action[0, :12]
        command_for_robot = pd_tau_targets_lcmt()
        self.joint_pos_target = (
            action[0, :12].detach().cpu().numpy()
            * self.cfg["env"]["control"]["actionScale"]
        ).flatten()
        self.joint_pos_target[[0, 3, 6, 9]] *= self.cfg["env"]["control"][
            "hipAddtionalScale"
        ]
        self.joint_pos_target = self.joint_pos_target
        self.joint_pos_target += self.default_dof_pos
        joint_pos_target = self.joint_pos_target[self.joint_idxs]
        self.joint_vel_target = np.zeros(12)
        # print(f'cjp {self.joint_pos_target}')

        command_for_robot.q_des = joint_pos_target
        command_for_robot.qd_des = self.joint_vel_target
        command_for_robot.kp = self.p_gains
        command_for_robot.kd = self.d_gains
        command_for_robot.tau_ff = np.zeros(12)
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10**6)
        command_for_robot.id = 0

        if hard_reset:
            command_for_robot.id = -1

        self.torques = (self.joint_pos_target - self.dof_pos) * self.p_gains + (
            self.joint_vel_target - self.dof_vel
        ) * self.d_gains

        lc.publish("pd_plustau_targets", command_for_robot.encode())

    def reset(self):
        self.actions = torch.zeros(12)
        self.time = time.time()
        self.timestep = 0
        return self.get_obs()

    def reset_gait_indices(self):
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float)

    def step_once(self, actions, hard_reset=False):
        clip_actions = self.cfg["env"]["clipActions"]
        self.last_actions = self.actions[:]
        self.actions = torch.clip(actions[0:1, :], -clip_actions, clip_actions)
        self.publish_action(self.actions, hard_reset=hard_reset)
        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.timestep % 100 == 0:
            print(f"frq: {1 / (time.time() - self.time)} Hz")
        self.time = time.time()
        obs = self.get_obs()

        self.timestep += 1
        return obs, None, None
