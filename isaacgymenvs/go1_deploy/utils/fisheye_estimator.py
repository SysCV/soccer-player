import torch
import math


class BallEstimator:
    def __init__(self, T1, T2, last_state=torch.tensor([0.0, 0.0, 0.0])):
        self.T1 = torch.tensor(T1)  # Transformation matrix for camera 1 (4x4)
        self.T2 = torch.tensor(T2)  # Transformation matrix for camera 2 (4x4)
        self.history_cam1_ground = None
        self.history_cam1_size = None
        self.history_cam2_ground = None
        self.history_cam2_size = None
        self.last_state = last_state
        self.f = 270.0
        self.cx_head = 4.372457397148538e02
        self.cy_head = 4.158324610639590e02
        self.cx_body = 4.632452480148337e02
        self.cy_body = 4.080649298750998e02
        self.ball_radius = 0.1
        self.head_y = 0.21675  # head 0.01675(cam) + 0.3(body) - 0.1 = 0.21675
        self.body_z = 0.148  # head -0.052(cam) + 0.3(body) - 0.1 = 0.148

    def update_cam1(self, xmin=None, ymin=None, xmax=None, ymax=None):
        if (
            xmin is not None
            and ymin is not None
            and xmax is not None
            and ymax is not None
        ):
            x, y, z = self.estimate1_3D_from_size(xmin, ymin, xmax, ymax)
            self.history_cam1_size = self.T1 @ torch.tensor(
                [[x], [y], [z], [1]]
            )  # Overwrite the history with the new measurement

            x, y, z = self.estimate1_3D_from_ground(xmin, ymin, xmax, ymax)
            self.history_cam1_ground = self.T1 @ torch.tensor(
                [[x], [y], [z], [1]]
            )  # Overwrite the history with the new measurement

    def update_cam2(self, xmin=None, ymin=None, xmax=None, ymax=None):
        if (
            xmin is not None
            and ymin is not None
            and xmax is not None
            and ymax is not None
        ):
            x, y, z = self.estimate2_3D_from_size(xmin, ymin, xmax, ymax)
            self.history_cam2_size = self.T2 @ torch.tensor([[x], [y], [z], [1]])

            x, y, z = self.estimate2_3D_from_ground(xmin, ymin, xmax, ymax)
            self.history_cam2_ground = self.T2 @ torch.tensor([[x], [y], [z], [1]])

    def estimate1_3D_from_size(self, xmin, ymin, xmax, ymax):
        px, py = self.get_box_center(xmin, ymin, xmax, ymax)
        ball_range_pixel = self.get_ball_range(xmin, ymin, xmax, ymax)
        ball_pixel_x = px - self.cx_head
        ball_pixel_y = py - self.cy_head
        r_pixel = math.sqrt(ball_pixel_x**2 + ball_pixel_y**2)
        phi_center = r_pixel / self.f
        phi_ball_half = ball_range_pixel / self.f / 2  # cloud be posi - (- posi)
        distance = self.ball_radius / math.sin(phi_ball_half)

        r = math.sin(phi_center) * distance

        x = (ball_pixel_x / r_pixel) * r
        y = (ball_pixel_y / r_pixel) * r
        z = math.cos(phi_center) * distance  # bug here
        # z = math.cos(theta) * r

        return x, y, z

    def estimate2_3D_from_size(self, xmin, ymin, xmax, ymax):
        px, py = self.get_box_center(xmin, ymin, xmax, ymax)
        ball_range_pixel = self.get_ball_range(xmin, ymin, xmax, ymax)
        ball_pixel_x = px - self.cx_body
        ball_pixel_y = py - self.cy_body
        r_pixel = math.sqrt(ball_pixel_x**2 + ball_pixel_y**2)
        phi_center = r_pixel / self.f
        phi_ball_half = ball_range_pixel / self.f / 2  # cloud be posi - (- posi)
        distance = self.ball_radius / math.sin(phi_ball_half)

        r = math.sin(phi_center) * distance

        x = (ball_pixel_x / r_pixel) * r
        y = (ball_pixel_y / r_pixel) * r
        z = math.cos(phi_center) * distance  # bug here
        # z = math.cos(theta) * r

        return x, y, z

    def estimate1_3D_from_ground(self, xmin, ymin, xmax, ymax):
        px, py = self.get_box_center(xmin, ymin, xmax, ymax)
        ball_pixel_x = px - self.cx_head
        ball_pixel_y = py - self.cy_head
        r_pixel = math.sqrt(ball_pixel_x**2 + ball_pixel_y**2)
        phi_center = r_pixel / self.f

        y = self.head_y
        x = (ball_pixel_x / ball_pixel_y) * y

        r = (r_pixel / ball_pixel_y) * y
        z = r / math.tan(phi_center)

        return x, y, z

    def estimate2_3D_from_ground(self, xmin, ymin, xmax, ymax):
        px, py = self.get_box_center(xmin, ymin, xmax, ymax)
        ball_pixel_x = px - self.cx_body
        ball_pixel_y = py - self.cy_body
        r_pixel = math.sqrt(ball_pixel_x**2 + ball_pixel_y**2)
        phi_center = r_pixel / self.f

        z = self.body_z

        r = math.tan(phi_center) * z

        x = (ball_pixel_x / r_pixel) * r
        y = (ball_pixel_y / r_pixel) * r

        return x, y, z

    def get_ball_range(self, xmin, ymin, xmax, ymax):
        return math.sqrt((xmax - xmin) * (ymax - ymin))

    def get_box_center(self, xmin, ymin, xmax, ymax):
        return (xmin + xmax) / 2, (ymin + ymax) / 2

    def pixel_r_to_phi(self, fx, fy, cx, cy, px, py):
        theta = math.sqrt(((px - cx) / fx) ** 2 + ((py - cy) / fy) ** 2)  # bug here!!
        return theta

    def get_estimation_result(self, sep_print=False):
        if self.history_cam1_size is not None and self.history_cam2_size is not None:
            # average of the four
            result = (
                self.history_cam1_size
                + self.history_cam2_size
                + self.history_cam1_ground
                + self.history_cam2_ground
            ) / 4
            self.last_state = result
        elif self.history_cam1_size is not None:
            result = (self.history_cam1_ground + self.history_cam1_size) / 2
            self.last_state = result
        elif self.history_cam2_size is not None:
            result = (self.history_cam2_ground + self.history_cam2_size) / 2
            self.last_state = result
        else:
            result = self.last_state

        if sep_print:
            if self.history_cam1_size is not None:
                print("history_cam1_size: ", self.history_cam1_size.squeeze())
            if self.history_cam2_size is not None:
                print("history_cam2_size: ", self.history_cam2_size.squeeze())
            if self.history_cam1_ground is not None:
                print("history_cam1_ground: ", self.history_cam1_ground.squeeze())
            if self.history_cam2_ground is not None:
                print("history_cam2_ground: ", self.history_cam2_ground.squeeze())

        self.history_cam1_size = None
        self.history_cam2_size = None
        self.history_cam1_ground = None
        self.history_cam2_ground = None

        return result.squeeze()
