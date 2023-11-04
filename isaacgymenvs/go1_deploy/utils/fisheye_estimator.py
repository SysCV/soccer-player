import torch
import math


class BallEstimator:
    def __init__(self, T1, T2, last_state=torch.tensor([0.0, 0.0, 0.0])):
        self.T1 = torch.tensor(T1)  # Transformation matrix for camera 1 (4x4)
        self.T2 = torch.tensor(T2)  # Transformation matrix for camera 2 (4x4)
        self.history_cam1 = None
        self.history_cam2 = None
        self.last_state = last_state

    def update_cam1(self, xmin=None, ymin=None, xmax=None, ymax=None):
        if (
            xmin is not None
            and ymin is not None
            and xmax is not None
            and ymax is not None
        ):
            x, y, z = self.estimate_3D_point(xmin, ymin, xmax, ymax)
            self.history_cam1 = self.T1 @ torch.tensor(
                [[x], [y], [z], [1]]
            )  # Overwrite the history with the new measurement

    def update_cam2(self, xmin=None, ymin=None, xmax=None, ymax=None):
        if (
            xmin is not None
            and ymin is not None
            and xmax is not None
            and ymax is not None
        ):
            x, y, z = self.estimate_3D_point(xmin, ymin, xmax, ymax)
            self.history_cam2 = self.T2 @ torch.tensor(
                [[x], [y], [z], [1]]
            )  # Overwrite the history with the new measurement

    def estimate_3D_point(self, xmin, ymin, xmax, ymax):
        fx = 3.7513105535330789e02
        fy = 3.8594489809074821e02
        cx = 4.3923707519180203e02
        cy = 4.0432640383775964e02
        phi_real_ball = 0.20

        px, py = self.get_box_center(xmin, ymin, xmax, ymax)
        x_range, y_range = self.get_box_range(xmin, ymin, xmax, ymax)
        theta = self.get_pixel_theta(fx, fy, cx, cy, px, py)
        distance = (fx * phi_real_ball / x_range) / 2 + (
            fy * phi_real_ball / y_range
        ) / 2
        r = math.sin(theta) * distance
        x_pixel = px - cx
        y_pixel = py - cy
        x = x_pixel / math.sqrt(x_pixel * x_pixel + y_pixel * y_pixel) * r
        y = y_pixel / math.sqrt(x_pixel * x_pixel + y_pixel * y_pixel) * r
        z = math.cos(theta) * r

        return x, y, z

    def get_box_range(self, xmin, ymin, xmax, ymax):
        return xmax - xmin, ymax - ymin

    def get_box_center(self, xmin, ymin, xmax, ymax):
        return (xmin + xmax) / 2, (ymin + ymax) / 2

    def get_pixel_theta(self, fx, fy, cx, cy, px, py):
        theta = math.sqrt(((px - cx) / fx) ** 2 + ((py - cy) / fy) ** 2)
        return theta

    def get_estimation_result(self, sep_print=False):
        if self.history_cam1 is not None and self.history_cam2 is not None:
            result = (self.history_cam1 + self.history_cam2) / 2
            self.last_state = result
        elif self.history_cam1 is not None:
            result = self.history_cam1
            self.last_state = result
        elif self.history_cam2 is not None:
            result = self.history_cam2
            self.last_state = result
        else:
            result = self.last_state

        if sep_print:
            if self.history_cam1 is not None:
                print("history_cam1: ", self.history_cam1.squeeze())
            if self.history_cam2 is not None:
                print("history_cam2: ", self.history_cam2.squeeze())

        self.history_cam1 = None
        self.history_cam2 = None

        return result.squeeze()
