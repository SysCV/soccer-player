import torch

pt_file = torch.load(
    "/home/gymuser/IsaacGymEnvs-main/isaacgymenvs/runs/dribble-PID_27-12-41-04/nn/dribble-PID.pth"
)


for k, v in pt_file.items():
    print(k, v)
