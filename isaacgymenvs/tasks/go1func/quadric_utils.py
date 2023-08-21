import torch
import isaacgymenvs.utils.torch_jit_utils as torch_jit_utils

# r = 3
# Qstar_element = torch.diag(torch.tensor([1.,1.,1.,-r**2]))

# dual_balls = Qstar_element.repeat(400,1,1)

def calc_projected_bbox(dual_balls, Quatw_root, pw_cam, K, pw_q):
    """
    dual_balls: (B, 4, 4)
    Quatw_c: (B, 4)
    pw_c: (B, 3)
    K: (3, 3)
    pw_b: (B, 3)
    """

    Tw_b = torch.cat([torch.eye(3, device=dual_balls.device).repeat(pw_q.shape[0], 1, 1), pw_q.unsqueeze(-1)], dim=-1)
    Tw_b = torch.cat([Tw_b, torch.tensor([0., 0., 0., 1.], device=Tw_b.device, dtype=Tw_b.dtype).unsqueeze(0).repeat(Tw_b.shape[0], 1, 1)], dim=-2)
    
    dual_balls_w = torch.matmul(Tw_b, torch.matmul(dual_balls, Tw_b.transpose(-1, -2)))

    Rr_c = torch.tensor([[0, 0, 1],
                          [-1, 0, 0],
                          [0, -1, 0]],
                          dtype=torch.float32, device=dual_balls.device)
    # Rw_r = torch_jit_utils.quaternion_to_matrix(Quatw_r)
    Rw_r = quaternions_to_rotation_matrices(Quatw_root)
    Rw_c = torch.matmul(Rw_r, Rr_c)
    Tw_c = torch.cat([Rw_c, pw_cam.unsqueeze(-1)], dim=-1)
    Tw_c = torch.cat([Tw_c, torch.tensor([0., 0., 0., 1.], device=Tw_c.device, dtype=Tw_c.dtype).unsqueeze(0).repeat(Tw_c.shape[0], 1, 1)], dim=-2)

    # print("ball in world...", Tw_b[0])

    # print("K...", K)

    
    Tc_w = torch.inverse(Tw_c)

    # print("world in cam", Tc_w[:, :3, :4][0])
    
    p34 = Tc_w[:, :3, :4]

    Gstar = torch.matmul(K, torch.matmul(p34, torch.matmul(dual_balls_w, torch.matmul(p34.transpose(-1, -2), K.transpose(-1, -2)))))

    sq_term_x = Gstar[..., 0, 2].square() - Gstar[..., 0, 0] * Gstar[..., 2, 2]
    sq_term_y = Gstar[..., 1, 2].square() - Gstar[..., 1, 1] * Gstar[..., 2, 2]

    # print("dual_balls in world...", dual_balls_w[0])
    # print("p34..", p34[0])
    # print("Gstar...", Gstar[0])
    # print("sq_term...", sq_term_x[0])
    
    assert (sq_term_x >= 0).all()
    assert (sq_term_y >= 0).all()

    xmin = (Gstar[..., 0, 2] + sq_term_x.sqrt()) / Gstar[..., 2, 2]
    ymin = (Gstar[..., 1, 2] + sq_term_y.sqrt()) / Gstar[..., 2, 2]
    xmax = (Gstar[..., 0, 2] - sq_term_x.sqrt()) / Gstar[..., 2, 2]
    ymax = (Gstar[..., 1, 2] - sq_term_y.sqrt()) / Gstar[..., 2, 2]

    y = torch.stack([xmin, ymin, xmax, ymax], dim=1) 

    # print(y[0])
    return y

# quat: x y z w
def quaternions_to_rotation_matrices(quaternions):
    # Normalize the quaternions
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)
    
    q0, q1, q2, q3 = quaternions[..., 3], quaternions[..., 0], quaternions[..., 1], quaternions[..., 2]

    rotation_matrices = torch.zeros(quaternions.shape[:-1] + (3, 3), device=quaternions.device)

    rotation_matrices[..., 0, 0] = 1 - 2 * (q2 ** 2 + q3 ** 2)
    rotation_matrices[..., 0, 1] = 2 * (q1 * q2 - q0 * q3)
    rotation_matrices[..., 0, 2] = 2 * (q1 * q3 + q0 * q2)

    rotation_matrices[..., 1, 0] = 2 * (q1 * q2 + q0 * q3)
    rotation_matrices[..., 1, 1] = 1 - 2 * (q1 ** 2 + q3 ** 2)
    rotation_matrices[..., 1, 2] = 2 * (q2 * q3 - q0 * q1)

    rotation_matrices[..., 2, 0] = 2 * (q1 * q3 - q0 * q2)
    rotation_matrices[..., 2, 1] = 2 * (q2 * q3 + q0 * q1)
    rotation_matrices[..., 2, 2] = 1 - 2 * (q1 ** 2 + q2 ** 2)

    return rotation_matrices

def add_bbox_on_numpy_img(img, xmin, ymin, xmax, ymax, box_color=(255, 255, 255), line_thickness=2):
    img[ymin:ymax, xmin:xmin+line_thickness] = box_color
    img[ymin:ymax, xmax-line_thickness:xmax] = box_color
    img[ymin:ymin+line_thickness, xmin:xmax] = box_color
    img[ymax-line_thickness:ymax, xmin:xmax] = box_color
    return img

def convert_bbox_to_img_coord(xmin, ymin, xmax, ymax, image_width, image_height, size_tolerance=5):
    # in order xmin, ymin, xmax, ymax
    xmin = min(max(0, int(xmin)), image_width-1)
    ymin = min(max(0, int(ymin)), image_height-1)
    xmax = max(0,min(image_width-1, int(xmax)))
    ymax = max(0,min(image_height-1, int(ymax)))

    valid = True if xmin + 2 < xmax and ymin + 2 < ymax else False

    return xmin, ymin, xmax, ymax

def cam_outside_q():
    pass

def cam_behind_q():
    pass

