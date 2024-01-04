import yaml
import matplotlib.pyplot as plt


# Function to load the trajectory data from a YAML file
def load_trajectory_from_yaml(file_path, env, method):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    trajectory_data = data[env][method]["trajectory"]
    return trajectory_data


# Function to plot the trajectory for different methods in the same environment
def plot_trajectory_comparison(file_path, env_names, pt_names, colors):
    # include index in loop iterating over env_names
    for i, env in enumerate(env_names):
        plt.figure(i, figsize=(8, 6))
        for pt, color in zip(pt_names, colors):
            trajectory_data = load_trajectory_from_yaml(file_path, env, pt)
            px = trajectory_data["px"]
            py = trajectory_data["py"]

            bx = trajectory_data["bx"]
            by = trajectory_data["by"]
            plt.plot(px, py, "-", color=color, label=f"Robot Trajectory {pt}")
            plt.plot(bx, by, "--", color=color, label=f"Ball Trajectory {pt}")

        plt.title(f"Trajectory Comparison in {env} Environment")
        plt.xlabel("Position X (m)")
        plt.ylabel("Position Y (m)")
        plt.grid(True)
        plt.legend()
    plt.show()


# Function to load the trajectory data from a YAML file
def load_trajectory_from_yaml(file_path, env, method):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    trajectory_data = data[env][method]["trajectory"]
    return trajectory_data


# Function to plot the trajectory
def plot_trajectory(trajectory_data):
    time = trajectory_data["time"]
    px = trajectory_data["px"]
    py = trajectory_data["py"]

    bx = trajectory_data["bx"]
    by = trajectory_data["by"]

    plt.figure(figsize=(8, 6))
    plt.plot(px, py, "-", label="Robot Trajectory")
    plt.plot(bx, by, "--", label="Ball Trajectory")
    plt.title("Trajectory Plot")
    plt.xlabel("Position X (m)")
    plt.ylabel("Position Y (m)")
    plt.grid(True)
    plt.legend()
    plt.show()


# File path and details
file_path = "./script/traj_plot_data.yaml"
env_names = ["zero", "mid"]
pt_names = ["baseline", "PID"]
colors = ["orange", "blue"]

# Load and plot the trajectory
plot_trajectory_comparison(file_path, env_names, pt_names, colors)
