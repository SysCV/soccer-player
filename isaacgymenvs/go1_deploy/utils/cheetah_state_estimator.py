import math
import select
import threading
import time
import queue

import numpy as np
from ultralytics import YOLO
import torch
from gi.repository import Gst
import cv2

from isaacgymenvs.go1_deploy.lcm_types.leg_control_data_lcmt import (
    leg_control_data_lcmt,
)
from isaacgymenvs.go1_deploy.lcm_types.rc_command_lcmt import rc_command_lcmt
from isaacgymenvs.go1_deploy.lcm_types.state_estimator_lcmt import state_estimator_lcmt
from isaacgymenvs.go1_deploy.lcm_types.noise_lcmt import noise_lcmt
from isaacgymenvs.go1_deploy.lcm_types.camera_message_lcmt import camera_message_lcmt
from isaacgymenvs.go1_deploy.lcm_types.camera_message_rect_wide import (
    camera_message_rect_wide,
)
from isaacgymenvs.go1_deploy.utils.gst_receiver import Video
from isaacgymenvs.go1_deploy.utils.fisheye_estimator import BallEstimator


def get_rpy_from_quaternion(q):
    w, x, y, z = q
    r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    p = np.arcsin(2 * (w * y - z * x))
    y = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([r, p, y])


def get_rotation_matrix_from_rpy(rpy):
    """
    Get rotation matrix from the given quaternion.
    Args:
        q (np.array[float[4]]): quaternion [w,x,y,z]
    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    r, p, y = rpy
    R_x = np.array(
        [[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]]
    )

    R_y = np.array(
        [[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]]
    )

    R_z = np.array(
        [[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]]
    )

    rot = np.dot(R_z, np.dot(R_y, R_x))
    return rot


class StateEstimator:
    def __init__(
        self, lc, use_cameras=True, use_gst=False, double_cam=False, device="cuda:0"
    ):
        # reverse legs
        self.double_cam = double_cam
        self.joint_idxs = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.contact_idxs = [1, 0, 3, 2]
        # self.joint_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.lc = lc

        self.rms_queue = queue.Queue()

        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.tau_est = np.zeros(12)
        self.world_lin_vel = np.zeros(3)
        self.world_ang_vel = np.zeros(3)
        self.euler = np.zeros(3)
        self.R = np.eye(3)
        self.buf_idx = 0

        self.smoothing_length = 12
        self.deuler_history = np.zeros((self.smoothing_length, 3))
        self.dt_history = np.zeros((self.smoothing_length, 1))
        self.euler_prev = np.zeros(3)
        self.timuprev = time.time()

        self.body_lin_vel = np.zeros(3)
        self.body_ang_vel = np.zeros(3)
        self.smoothing_ratio = 0.2

        self.contact_state = np.ones(4)
        self.contact_state_air = np.array([200, 50, 200, 50])  # self calibration

        self.mode = 0
        self.ctrlmode_left = 0
        self.ctrlmode_right = 0
        self.left_stick = [0, 0]
        self.right_stick = [0, 0]
        self.left_upper_switch = 0
        self.left_lower_left_switch = 0
        self.left_lower_right_switch = 0
        self.right_upper_switch = 0
        self.right_lower_left_switch = 0
        self.right_lower_right_switch = 0
        self.left_upper_switch_pressed = 0
        self.left_lower_left_switch_pressed = 0
        self.left_lower_right_switch_pressed = 0
        self.right_upper_switch_pressed = 0
        self.right_lower_left_switch_pressed = 0
        self.right_lower_right_switch_pressed = 0

        # default trotting gait
        self.cmd_freq = 3.0
        self.cmd_phase = 0.5
        self.cmd_offset = 0.0
        self.cmd_duration = 0.5

        self.init_time = time.time()
        self.received_first_legdata = False

        self.imu_subscription = self.lc.subscribe("state_estimator_data", self._imu_cb)
        self.legdata_state_subscription = self.lc.subscribe(
            "leg_control_data", self._legdata_cb
        )
        self.rc_command_subscription = self.lc.subscribe(
            "rc_command", self._rc_command_cb
        )
        self.noise_subscription = self.lc.subscribe("noise_data", self._noise_cb)

        if use_cameras:
            for cam_id in [1, 2, 3, 4, 5]:
                self.camera_subscription = self.lc.subscribe(
                    f"camera{cam_id}", self._camera_cb
                )
            self.camera_names = ["front", "bottom", "left", "right", "rear"]
            for cam_name in self.camera_names:
                self.camera_subscription = self.lc.subscribe(
                    f"rect_image_{cam_name}", self._rect_camera_cb
                )

        if use_gst:
            Gst.init(None)
            self.model = YOLO(
                "/home/gymuser/IsaacGymEnvs-main/isaacgymenvs/dataset/best_99.pt",
            )

            self.gst_video = Video(9200, 3)
            print("Initialising stream...")
            waited = 0
            while not self.gst_video.frame_available():
                waited += 1
                print(
                    "\r  Frame not available, need to wait 10 second (x{})".format(
                        waited
                    ),
                    end="",
                )
                time.sleep(1)
            print("\nSuccess!\nStarting streaming - ")
            self.gst_video.frame()

        if double_cam:
            # Example usage:
            T1 = [
                [0.0, 0.0, 1.0, 0.289],
                [-1.0, 0.0, 0.0, -0.0125],
                [0.0, -1.0, 0.0, 0.01675],
                [0.0, 0.0, 0.0, 1.0],
            ]

            T2 = [
                [0.0, -1.0, 0.0, -0.0825],
                [-1.0, 0.0, 0.0, -0.0125],
                [0.0, 0.0, -1.0, -0.052],
                [0.0, 0.0, 0.0, 1.0],
            ]

            self.estimator = BallEstimator(T1, T2)
            self.gst_video2 = Video(9209, 1)
            print("Initialising second stream...")
            waited = 0
            while not self.gst_video2.frame_available():
                waited += 1
                print(
                    "\r  Frame not available, need to wait 10 second (x{})".format(
                        waited
                    ),
                    end="",
                )
                time.sleep(1)
            print("\nSuccess!\nStarting streaming - ")
            self.gst_video2.frame()

        self.camera_image_left = None
        self.camera_image_right = None
        self.camera_image_front = None
        self.camera_image_bottom = None
        self.camera_image_rear = None

        self.body_loc = np.array([0, 0, 0])
        self.body_quat = np.array([0, 0, 0, 1])

    def get_body_linear_vel(self):
        self.body_lin_vel = np.dot(self.R.T, self.world_lin_vel)
        return self.body_lin_vel

    def get_body_angular_vel(self):
        self.body_ang_vel = (
            self.smoothing_ratio
            * np.mean(self.deuler_history / self.dt_history, axis=0)
            + (1 - self.smoothing_ratio) * self.body_ang_vel
        )
        return self.body_ang_vel

    def get_gravity_vector(self):
        grav = np.dot(self.R.T, np.array([0, 0, -1]))
        return grav

    def get_contact_state(self):
        return self.contact_state[self.contact_idxs]

    def get_rpy(self):
        return self.euler

    def get_command(self):
        MODES_LEFT = ["body_height", "lat_vel", "stance_width"]
        MODES_RIGHT = ["step_frequency", "footswing_height", "body_pitch"]

        if self.left_upper_switch_pressed:
            self.ctrlmode_left = (self.ctrlmode_left + 1) % 3
            self.left_upper_switch_pressed = False
        if self.right_upper_switch_pressed:
            self.ctrlmode_right = (self.ctrlmode_right + 1) % 3
            self.right_upper_switch_pressed = False

        # MODE_LEFT = MODES_LEFT[self.ctrlmode_left]
        MODE_LEFT = MODES_LEFT[1]  # here fix the mode left
        MODE_RIGHT = MODES_RIGHT[self.ctrlmode_right]

        # always in use
        cmd_x = 1 * self.left_stick[1]
        cmd_yaw = -1 * self.right_stick[0]

        # default values
        cmd_y = 0.0  # -1 * self.left_stick[0]
        cmd_height = 0.0
        cmd_footswing = 0.08
        cmd_stance_width = 0.33
        cmd_stance_length = 0.40
        cmd_ori_pitch = 0.0
        cmd_ori_roll = 0.0
        cmd_freq = 3.0

        # joystick commands
        if MODE_LEFT == "body_height":
            cmd_height = 0.3 * self.left_stick[0]
        elif MODE_LEFT == "lat_vel":
            cmd_y = -1.0 * self.left_stick[0]  # here y is inversed from controller
        elif MODE_LEFT == "stance_width":
            cmd_stance_width = 0.275 + 0.175 * self.left_stick[0]
        if MODE_RIGHT == "step_frequency":
            min_freq = 2.0
            max_freq = 4.0
            cmd_freq = (1 + self.right_stick[1]) / 2 * (max_freq - min_freq) + min_freq
        elif MODE_RIGHT == "footswing_height":
            cmd_footswing = max(0, self.right_stick[1]) * 0.32 + 0.03
        elif MODE_RIGHT == "body_pitch":
            cmd_ori_pitch = -0.4 * self.right_stick[1]

        # gait buttons
        if self.mode == 0:
            self.cmd_phase = 0.5
            self.cmd_offset = 0.0
            self.cmd_bound = 0.0
            self.cmd_duration = 0.5
        elif self.mode == 1:
            self.cmd_phase = 0.0
            self.cmd_offset = 0.0
            self.cmd_bound = 0.0
            self.cmd_duration = 0.5
        elif self.mode == 2:
            self.cmd_phase = 0.0
            self.cmd_offset = 0.5
            self.cmd_bound = 0.0
            self.cmd_duration = 0.5
        elif self.mode == 3:
            self.cmd_phase = 0.0
            self.cmd_offset = 0.0
            self.cmd_bound = 0.5
            self.cmd_duration = 0.5
        else:
            self.cmd_phase = 0.5
            self.cmd_offset = 0.0
            self.cmd_bound = 0.0
            self.cmd_duration = 0.5

        return np.array(
            [
                cmd_x,
                cmd_y,
                cmd_yaw,
                cmd_height,
                cmd_freq,
                self.cmd_phase,
                self.cmd_offset,
                self.cmd_bound,
                self.cmd_duration,
                cmd_footswing,
                cmd_ori_pitch,
                cmd_ori_roll,
                cmd_stance_width,
                cmd_stance_length,
                0,
                0,
                0,
                0,
                0,
            ]
        )

    def get_buttons(self):
        return np.array(
            [
                self.left_lower_left_switch,
                self.left_upper_switch,
                self.right_lower_right_switch,
                self.right_upper_switch,
            ]
        )

    def get_rms(self):
        """Calculate the average 'rms' from the queue and clear it."""
        if self.rms_queue.empty():
            return 0  # Return None if the queue is empty

        sum_of_squares = 0
        count = 0

        # Calculate the sum of squares and count of 'rms' values in the queue
        while not self.rms_queue.empty():
            rms = self.rms_queue.get()
            sum_of_squares += rms * rms
            count += 1

        # Calculate the average 'rms'
        average_rms = math.sqrt(sum_of_squares / count)

        print(" msg in queue: ", count, "/10", end=" ")
        return average_rms

    def get_dof_pos(self):
        # print("dofposquery", self.joint_pos[self.joint_idxs])
        return self.joint_pos[self.joint_idxs]

    def get_dof_vel(self):
        return self.joint_vel[self.joint_idxs]

    def get_tau_est(self):
        return self.tau_est[self.joint_idxs]

    def get_yaw(self):
        return self.euler[2]

    def get_body_loc(self):
        return np.array(self.body_loc)

    def get_body_quat(self):
        return np.array(self.body_quat)

    def get_camera_front(self):
        return self.camera_image_front

    def get_camera_bottom(self):
        return self.camera_image_bottom

    def get_camera_rear(self):
        return self.camera_image_rear

    def get_camera_left(self):
        return self.camera_image_left

    def get_camera_right(self):
        return self.camera_image_right

    def get_camera_gst(self):
        return self.gst_video.frame()

    def get_camera_gst2(self):
        assert self.double_cam, "Double camera not enabled"
        return self.gst_video2.frame()

    def _legdata_cb(self, channel, data):
        # print("update legdata")
        if not self.received_first_legdata:
            self.received_first_legdata = True

            print(f"First legdata: {time.time() - self.init_time}")
            print("=== First legdata received from LCM ===")

        msg = leg_control_data_lcmt.decode(data)
        # print(msg.q)
        self.joint_pos = np.array(msg.q)
        # print(self.joint_pos)
        self.joint_vel = np.array(msg.qd)
        self.tau_est = np.array(msg.tau_est)
        # print(f"update legdata {msg.id}")

    def _noise_cb(self, channel, data):
        # print("update noise")
        msg = noise_lcmt.decode(data)
        self.rms_queue.put(msg.rms)

    def _imu_cb(self, channel, data):
        # print("update imu")
        msg = state_estimator_lcmt.decode(data)

        self.euler = np.array(msg.rpy)
        # print(self.euler)

        self.R = get_rotation_matrix_from_rpy(self.euler)

        self.contact_state = 1.0 * (
            np.array(msg.contact_estimate) > self.contact_state_air
        )
        # print(np.array(msg.contact_estimate))
        self.deuler_history[self.buf_idx % self.smoothing_length, :] = (
            msg.rpy - self.euler_prev
        )
        self.dt_history[self.buf_idx % self.smoothing_length] = (
            time.time() - self.timuprev
        )

        self.timuprev = time.time()

        self.buf_idx += 1
        self.euler_prev = np.array(msg.rpy)

    def _sensor_cb(self, channel, data):
        pass

    def _rc_command_cb(self, channel, data):
        msg = rc_command_lcmt.decode(data)

        self.left_upper_switch_pressed = (
            msg.left_upper_switch and not self.left_upper_switch
        ) or self.left_upper_switch_pressed
        self.left_lower_left_switch_pressed = (
            msg.left_lower_left_switch and not self.left_lower_left_switch
        ) or self.left_lower_left_switch_pressed
        self.left_lower_right_switch_pressed = (
            msg.left_lower_right_switch and not self.left_lower_right_switch
        ) or self.left_lower_right_switch_pressed
        self.right_upper_switch_pressed = (
            msg.right_upper_switch and not self.right_upper_switch
        ) or self.right_upper_switch_pressed
        self.right_lower_left_switch_pressed = (
            msg.right_lower_left_switch and not self.right_lower_left_switch
        ) or self.right_lower_left_switch_pressed
        self.right_lower_right_switch_pressed = (
            msg.right_lower_right_switch and not self.right_lower_right_switch
        ) or self.right_lower_right_switch_pressed

        self.mode = msg.mode
        self.right_stick = msg.right_stick
        self.left_stick = msg.left_stick
        self.left_upper_switch = msg.left_upper_switch
        self.left_lower_left_switch = msg.left_lower_left_switch
        self.left_lower_right_switch = msg.left_lower_right_switch
        self.right_upper_switch = msg.right_upper_switch
        self.right_lower_left_switch = msg.right_lower_left_switch
        self.right_lower_right_switch = msg.right_lower_right_switch

        # print(self.right_stick, self.left_stick)

    def _camera_cb(self, channel, data):
        msg = camera_message_lcmt.decode(data)

        img = np.fromstring(msg.data, dtype=np.uint8)
        img = img.reshape((3, 200, 464)).transpose(1, 2, 0)

        cam_id = int(channel[-1])
        if cam_id == 1:
            self.camera_image_front = img
        elif cam_id == 2:
            self.camera_image_bottom = img
        elif cam_id == 3:
            self.camera_image_left = img
        elif cam_id == 4:
            self.camera_image_right = img
        elif cam_id == 5:
            self.camera_image_rear = img
        else:
            print("Image received from camera with unknown ID#!")

        # im = Image.fromarray(img).convert('RGB')

        # im.save("test_image_" + channel + ".jpg")
        # print(channel)

    def _rect_camera_cb(self, channel, data):
        message_types = [
            camera_message_rect_wide,
            camera_message_rect_wide,
            camera_message_rect_wide,
            camera_message_rect_wide,
            camera_message_rect_wide,
        ]
        image_shapes = [
            (116, 100, 3),
            (116, 100, 3),
            (116, 100, 3),
            (116, 100, 3),
            (116, 100, 3),
        ]

        cam_name = channel.split("_")[-1]
        # print(f"received py from {cam_name}")
        cam_id = self.camera_names.index(cam_name) + 1

        msg = message_types[cam_id - 1].decode(data)

        img = np.fromstring(msg.data, dtype=np.uint8)
        img = np.flip(
            np.flip(
                img.reshape(
                    (
                        image_shapes[cam_id - 1][2],
                        image_shapes[cam_id - 1][1],
                        image_shapes[cam_id - 1][0],
                    )
                ),
                axis=0,
            ),
            axis=1,
        ).transpose(1, 2, 0)
        # print(img.shape)
        # img = np.flip(np.flip(img.reshape(image_shapes[cam_id - 1]), axis=0), axis=1)[:, :,
        #       [2, 1, 0]]  # .transpose(1, 2, 0)

        if cam_id == 1:
            self.camera_image_front = img
        elif cam_id == 2:
            self.camera_image_bottom = img
        elif cam_id == 3:
            self.camera_image_left = img
        elif cam_id == 4:
            self.camera_image_right = img
        elif cam_id == 5:
            self.camera_image_rear = img
        else:
            print("Image received from camera with unknown ID#!")

    def get_gst_frame_result(self):
        result = self.model(self.gst_video.frame())  # TODO: fix this
        return result

    def get_ball_estimation(self, plot_img=False, print_state=False, print_speed=False):
        assert self.double_cam, "Double camera not enabled"
        # pic_list = [self.gst_video.frame(), self.gst_video2.frame()]
        result = self.model(
            [self.gst_video.frame(), self.gst_video2.frame()],
            device="cuda:0",
            verbose=True,
        )

        r1 = list(result)[0]
        boxes1 = r1.boxes.xyxy
        confidences1 = r1.boxes.conf
        # pick the box with highest confidence higher than 0.5

        valid_indices1 = (confidences1 > 0.5).nonzero()
        if valid_indices1.size(0) > 0:
            _, max_index1 = torch.max(confidences1[valid_indices1], dim=0)

            # Get the corresponding box from the valid indices
            xyxy1 = boxes1[valid_indices1[max_index1]].squeeze()
            # print(xyxy1)
            self.estimator.update_cam1(
                xyxy1[0].item(),
                xyxy1[1].item(),
                xyxy1[2].item(),
                xyxy1[3].item(),
            )
        else:
            self.estimator.update_cam1()

        r2 = list(result)[1]
        boxes2 = r2.boxes.xyxy
        confidences2 = r2.boxes.conf
        # pick the box with highest confidence higher than 0.5

        valid_indices2 = (confidences2 > 0.5).nonzero()
        if valid_indices2.size(0) > 0:
            _, max_index2 = torch.max(confidences2[valid_indices2], dim=0)

            # Get the corresponding box from the valid indices
            xyxy2 = boxes2[valid_indices2[max_index2]].squeeze()
            self.estimator.update_cam2(
                xyxy2[0].item(),
                xyxy2[1].item(),
                xyxy2[2].item(),
                xyxy2[3].item(),
            )

        else:
            self.estimator.update_cam2()

        if plot_img:
            im_array1 = r1.plot()  # plot a BGR numpy array of predictions
            cv2.imshow("result1", im_array1)
            im_array2 = r2.plot()  # plot a BGR numpy array of predictions
            cv2.imshow("result2", im_array2)
            cv2.waitKey(1)

        if print_speed:
            print(r1.speed)
            print(r2.speed)

        estimated_position = self.estimator.get_estimation_result(sep_print=print_state)
        if print_state:
            print("fused pos:", estimated_position)
        return estimated_position

    def poll(self, cb=None):
        t = time.time()
        try:
            while True:
                timeout = 0.01
                rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
                if rfds:
                    # print("message received!")
                    self.lc.handle()
                    # print(f'Freq {1. / (time.time() - t)} Hz'); t = time.time()
                else:
                    continue
                    # print(f'waiting for message... Freq {1. / (time.time() - t)} Hz'); t = time.time()
                #    if cb is not None:
                #        cb()
        except KeyboardInterrupt:
            pass

    def spin(self):
        self.run_thread = threading.Thread(target=self.poll, daemon=False)
        self.run_thread.start()

    def close(self):
        self.lc.unsubscribe(self.legdata_state_subscription)


if __name__ == "__main__":
    import lcm

    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
    se = StateEstimator(lc)
    se.poll()
