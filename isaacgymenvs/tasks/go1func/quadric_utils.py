import torch
import isaacgymenvs.utils.torch_jit_utils as torch_jit_utils

# r = 3
# Qstar_element = torch.diag(torch.tensor([1.,1.,1.,-r**2]))

# dual_balls = Qstar_element.repeat(400,1,1)


def calc_projected_bbox(
    dual_element, Quatw_root, pw_root, K, pw_ball, proot_cam, facing_down=False
):
    """
    dual_balls: (B, 4, 4) \\
    Quatw_c: (B, 4) \\
    pw_c: (B, 3) \\
    K: (3, 3) \\
    pw_b: (B, 3) \\
    """

    Tw_ball = torch.cat(
        [
            torch.eye(3, device=pw_ball.device).repeat(pw_ball.shape[0], 1, 1),
            pw_ball.unsqueeze(-1),
        ],
        dim=-1,
    )
    Tw_ball = torch.cat(
        [
            Tw_ball,
            torch.tensor(
                [0.0, 0.0, 0.0, 1.0], device=Tw_ball.device, dtype=Tw_ball.dtype
            )
            .unsqueeze(0)
            .repeat(Tw_ball.shape[0], 1, 1),
        ],
        dim=-2,
    )

    Q_w = torch.matmul(Tw_ball, torch.matmul(dual_element, Tw_ball.transpose(-1, -2)))

    if facing_down:
        Rr_c = torch.tensor(
            [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
            dtype=torch.float32,
            device=pw_ball.device,
        )  # rot the cam in root place to make z axis the front axis
    else:
        Rr_c = torch.tensor(
            [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            dtype=torch.float32,
            device=pw_ball.device,
        )  # rot the cam in root place to make z axis the front axis
    Tr_c = torch.cat([Rr_c, proot_cam.unsqueeze(-1)], dim=-1)
    Tr_c = torch.cat(
        [Tr_c, torch.tensor([0, 0, 0, 1], device=Tr_c.device).unsqueeze(0)], dim=-2
    )
    # Rw_r = torch_jit_utils.quaternion_to_matrix(Quatw_r)
    Rw_r = quaternions_to_rotation_matrices(Quatw_root)
    Tw_r = torch.cat(
        [Rw_r, pw_root.unsqueeze(-1)],
        dim=-1,
    )
    Tw_r = torch.cat(
        [
            Tw_r,
            torch.tensor(
                [0.0, 0.0, 0.0, 1.0], device=Tw_ball.device, dtype=Tw_ball.dtype
            )
            .unsqueeze(0)
            .repeat(Tw_ball.shape[0], 1, 1),
        ],
        dim=-2,
    )
    Tw_c = torch.matmul(Tw_r, Tr_c)

    # print("ball in world...", Tw_b[0])

    # print("K...", K)

    Tc_w = torch.inverse(Tw_c)

    # print("world in cam", Tc_w[:, :3, :4][0])

    p34 = Tc_w[:, :3, :4]

    Gstar = torch.matmul(
        K,
        torch.matmul(
            p34,
            torch.matmul(Q_w, torch.matmul(p34.transpose(-1, -2), K.transpose(-1, -2))),
        ),
    )

    sq_term_x = Gstar[..., 0, 2].square() - Gstar[..., 0, 0] * Gstar[..., 2, 2]
    sq_term_y = Gstar[..., 1, 2].square() - Gstar[..., 1, 1] * Gstar[..., 2, 2]

    valid_index = (
        (sq_term_x >= 0)
        & (sq_term_y >= 0)
        & cam_outside_q(pw_ball, pw_root)
        & cam_behind_q(Tw_c, pw_ball)
    )

    # print("dual_balls in world...", dual_balls_w[0])
    # print("p34..", p34[0])
    # print("Gstar...", Gstar[0])
    # print("sq_term...", sq_term_x[0])

    # xmin, ymin, xmax, ymax
    bbox_results = torch.zeros(
        (pw_ball.shape[0], 4), device=pw_ball.device, dtype=pw_ball.dtype
    )

    bbox_results[valid_index, 0] = (
        Gstar[valid_index, 0, 2] + sq_term_x[valid_index].sqrt()
    ) / Gstar[valid_index, 2, 2]
    bbox_results[valid_index, 1] = (
        Gstar[valid_index, 1, 2] + sq_term_y[valid_index].sqrt()
    ) / Gstar[valid_index, 2, 2]
    bbox_results[valid_index, 2] = (
        Gstar[valid_index, 0, 2] - sq_term_x[valid_index].sqrt()
    ) / Gstar[valid_index, 2, 2]
    bbox_results[valid_index, 3] = (
        Gstar[valid_index, 1, 2] - sq_term_y[valid_index].sqrt()
    ) / Gstar[valid_index, 2, 2]

    # print("bbox_results...", bbox_results)

    # print(y[0])
    return bbox_results


# quat: x y z w
def quaternions_to_rotation_matrices(quaternions):
    # Normalize the quaternions
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)

    q0, q1, q2, q3 = (
        quaternions[..., 3],
        quaternions[..., 0],
        quaternions[..., 1],
        quaternions[..., 2],
    )

    rotation_matrices = torch.zeros(
        quaternions.shape[:-1] + (3, 3), device=quaternions.device
    )

    rotation_matrices[..., 0, 0] = 1 - 2 * (q2**2 + q3**2)
    rotation_matrices[..., 0, 1] = 2 * (q1 * q2 - q0 * q3)
    rotation_matrices[..., 0, 2] = 2 * (q1 * q3 + q0 * q2)

    rotation_matrices[..., 1, 0] = 2 * (q1 * q2 + q0 * q3)
    rotation_matrices[..., 1, 1] = 1 - 2 * (q1**2 + q3**2)
    rotation_matrices[..., 1, 2] = 2 * (q2 * q3 - q0 * q1)

    rotation_matrices[..., 2, 0] = 2 * (q1 * q3 - q0 * q2)
    rotation_matrices[..., 2, 1] = 2 * (q2 * q3 + q0 * q1)
    rotation_matrices[..., 2, 2] = 1 - 2 * (q1**2 + q2**2)

    return rotation_matrices


def add_bbox_on_numpy_img(
    img, xmin, ymin, xmax, ymax, box_color=(255, 255, 255), line_thickness=2
):
    img[ymin:ymax, xmin : xmin + line_thickness] = box_color
    img[ymin:ymax, xmax - line_thickness : xmax] = box_color
    img[ymin : ymin + line_thickness, xmin:xmax] = box_color
    img[ymax - line_thickness : ymax, xmin:xmax] = box_color
    return img


def convert_bbox_to_img_coord(xy_minmax, image_width, image_height, size_tolerance=0):
    pixel_bbox = torch.zeros_like(xy_minmax, dtype=torch.int32)
    # in order xmin, ymin, xmax, ymax
    pixel_bbox[:, [0, 2]] = torch.clamp(
        xy_minmax[:, [0, 2]].to(torch.int32), min=0, max=image_width - 1
    )
    pixel_bbox[:, [1, 3]] = torch.clamp(
        xy_minmax[:, [1, 3]].to(torch.int32), min=0, max=image_height - 1
    )

    valid_box = ((pixel_bbox[:, 0] + size_tolerance) < pixel_bbox[:, 2]) & (
        (pixel_bbox[:, 1] + size_tolerance) < pixel_bbox[:, 3]
    )

    pixel_bbox[~valid_box, :] = 0

    return pixel_bbox


def convert_bbox_to_01(xy_minmax, image_width, image_height, size_tolerance=0):
    pixel_bbox = torch.zeros_like(xy_minmax, dtype=torch.float32)
    # in order xmin, ymin, xmax, ymax
    pixel_bbox[:, [0, 2]] = torch.clamp(
        xy_minmax[:, [0, 2]], min=0, max=image_width - 1
    )
    pixel_bbox[:, [1, 3]] = torch.clamp(
        xy_minmax[:, [1, 3]], min=0, max=image_height - 1
    )

    valid_box = ((pixel_bbox[:, 0] + size_tolerance) < pixel_bbox[:, 2]) & (
        (pixel_bbox[:, 1] + size_tolerance) < pixel_bbox[:, 3]
    )

    pixel_bbox[~valid_box, :] = 0.0

    pixel_bbox[:, [0, 2]] = pixel_bbox[:, [0, 2]] / (image_width - 1)
    pixel_bbox[:, [1, 3]] = pixel_bbox[:, [1, 3]] / (image_height - 1)

    return pixel_bbox


def cam_outside_q(pw_q, pw_c, tolerance=0.1):
    # because we are using the simulator, we can assume that the camera is always outside the quadric
    return torch.norm(pw_c - pw_q, dim=-1) > tolerance


def cam_behind_q(Tw_c, pw_q):
    envs = pw_q.shape[0]
    pw_q_4 = torch.cat(
        [
            pw_q.unsqueeze(-1),
            torch.ones((envs, 1, 1), device=pw_q.device, dtype=pw_q.dtype),
        ],
        dim=-2,
    )
    pc_q = torch.matmul(Tw_c.inverse(), pw_q_4).squeeze(-1)
    return pc_q[:, 2] > 0
