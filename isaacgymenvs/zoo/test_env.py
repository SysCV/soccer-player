import time
import sys


def print_log_messages(log_messages):
    for message in log_messages:
        print(message)
        sys.stdout.flush()


def print_progress_bar(iterable, total, length=50):
    for i, item in enumerate(iterable):
        progress = (i + 1) / total
        bar_length = int(length * progress)
        bar = "#" * bar_length + "-" * (length - bar_length)
        sys.stdout.write(f"\r[{bar}] {int(progress * 100)}%")
        sys.stdout.flush()
        yield item


log_messages = ["Log message 1", "Log message 2", "Log message 3"]

# Print log messages above the progress bar
print_log_messages(log_messages)

# Simulate a task (e.g., a loop)
total_iterations = 100
for _ in print_progress_bar(range(total_iterations), total_iterations):
    time.sleep(0.1)

# Print a new line to separate from the progress bar
print("\nTask completed.")
# import isaacgym
# import isaacgymenvs
# import torch

# num_envs = 1

# envs = isaacgymenvs.make(
# 	seed=0,
# 	task="Ant",
# 	num_envs=num_envs,
# 	sim_device="cuda:2",
# 	rl_device="cuda:2",
# 	headless=False,
# 	force_render=False,
# 	graphics_device_id=0,
# )

# print("Observation space is", envs.observation_space)
# print("Action space is", envs.action_space)
# obs = envs.reset()
# for _ in range(20):
# 	random_actions = 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device = 'cuda:0') - 1.0
# 	envs.step(random_actions)
# 	print("Action space is", envs.action_space)
