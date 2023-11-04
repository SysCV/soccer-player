import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D
import torch
import math

# Constants
radius = 3.0  # Radius of the larger circular trajectory
angular_speed = 0.4 * math.pi  # Angular speed in radians per second

# PD Control Constants (2D vectors)
kp_pos = torch.tensor(
    [300.0, 300.0]
)  # Proportional gain for position control (x and y)
kd_pos = torch.tensor([0.1, 0.1])  # Derivative gain for position control (x and y)
kp_ang = 0.5  # Proportional gain for angular speed control
kd_ang = 0.0  # Derivative gain for angular speed control

# Initial state of the rectangle (2D vector)
rect_pos = torch.tensor([0.0, 0.0])
rect_linear_speed = torch.tensor([0.0, 0.0])  # Initial linear speed of the rectangle
rect_angle = 0.0  # Initial angle of the rectangle (radians)
rect_angular_speed = 0.0  # Initial angular speed of the rectangle

# Rectangle Dimensions (1:2 aspect ratio)
rect_length = 1.0
rect_width = 0.5

# Simulation Parameters
dt = 0.02  # Time step (50 Hz)

# Lists to store data for plotting
ball_positions = []
rect_positions = []
rect_linear_speeds = []
rect_angles = []
rect_angular_speeds = []

# Initialize previous values for the first frame
prev_error_pos = torch.zeros(2)
prev_error_ang = 0.0
prev_ball_pos = torch.zeros(2)


# Function to update the position and speed of the rectangle
def update_position(frame):
    global rect_pos, rect_linear_speed, rect_angle, rect_angular_speed, prev_error_pos, prev_error_ang, prev_ball_pos  # Declare variables as global

    # Calculate the angle for the circular motion
    angle = angular_speed * frame * dt

    # Calculate the new position of the ball (2D vector)
    ball_pos = torch.tensor([radius * math.cos(angle), radius * math.sin(angle)])

    # Calculate the error for position control (2D vector)
    error_pos = ball_pos - rect_pos

    # Calculate the error change (delta error)
    if frame > 0:
        delta_error_pos = error_pos - prev_error_pos
    else:
        delta_error_pos = torch.zeros(2)  # Initialize with zeros for the first frame

    # Calculate the ball's velocity vector
    ball_speed = (ball_pos - prev_ball_pos) / dt

    # Calculate the difference between the rectangle's speed and the ball's speed
    speed_diff = rect_linear_speed - ball_speed

    # Apply PD control to update the linear speed of the rectangle (2D vector)
    control_output_pos = kp_pos * error_pos - kd_pos * delta_error_pos - speed_diff

    # Update the linear speed of the rectangle
    rect_linear_speed = control_output_pos * dt

    # Calculate the desired angle to the ball
    target_angle = math.atan2(ball_pos[1] - rect_pos[1], ball_pos[0] - rect_pos[0])

    # Calculate the error for angular speed control
    error_ang = target_angle - rect_angle

    # Ensure the error is within the range [-pi, pi]
    if error_ang > math.pi:
        error_ang -= 2 * math.pi
    elif error_ang < -math.pi:
        error_ang += 2 * math.pi

    # Calculate the change in error for angular speed control
    delta_error_ang = error_ang - prev_error_ang

    # Apply PD control to update the angular speed of the rectangle
    control_output_ang = kp_ang * error_ang - kd_ang * delta_error_ang

    # Update the angular speed of the rectangle
    rect_angular_speed = control_output_ang

    # Update the position and angle of the rectangle
    rect_pos[:2] += rect_linear_speed * dt
    rect_angle += rect_angular_speed * dt

    # Keep rect_angle within the range [-pi, pi]
    if rect_angle > math.pi:
        rect_angle -= 2 * math.pi
    elif rect_angle < -math.pi:
        rect_angle += 2 * math.pi

    # Store previous values for the next frame
    prev_error_pos = error_pos
    prev_error_ang = error_ang
    prev_ball_pos = ball_pos

    # Append data for plotting
    ball_positions.append(
        ball_pos.numpy()
    )  # Convert the Torch tensor to a NumPy array for plotting
    rect_positions.append(
        rect_pos.numpy()
    )  # Convert the Torch tensor to a NumPy array for plotting
    rect_linear_speeds.append(
        rect_linear_speed.numpy()
    )  # Convert the Torch tensor to a NumPy array for plotting
    rect_angles.append(rect_angle)
    rect_angular_speeds.append(rect_angular_speed)

    # Clear the previous frame
    ax.clear()

    # Plot the ball at the new position
    ax.scatter(*ball_pos, marker="o", color="blue", s=100)

    # Plot the rotated rectangle (1:2 aspect ratio)
    rect_center_x = rect_pos[0]  # X-coordinate of the rectangle's center
    rect_center_y = rect_pos[1]  # Y-coordinate of the rectangle's center
    rect = plt.Rectangle(
        (rect_center_x - rect_width / 2, rect_center_y - rect_length / 2),
        rect_length,
        rect_width,
        transform=Affine2D().rotate_around(rect_center_x, rect_center_y, rect_angle),
        color="red",
    )
    ax.add_patch(rect)

    # Plot a line indicating the heading of the rectangle
    heading_length = 1.0
    heading_x = rect_center_x + heading_length * math.cos(rect_angle)
    heading_y = rect_center_y + heading_length * math.sin(rect_angle)
    ax.plot([rect_center_x, heading_x], [rect_center_y, heading_y], color="green")

    # Plot a line linking the rectangle and the ball
    ax.plot(
        [rect_center_x, ball_pos[0]],
        [rect_center_y, ball_pos[1]],
        linestyle="--",
        color="purple",
    )

    # Set axis limits
    ax.set_xlim(-5, 5)  # Adjust these limits as needed
    ax.set_ylim(-5, 5)


# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Create an animation
ani = animation.FuncAnimation(fig, update_position, blit=False, interval=int(dt * 1000))

# Show the animation
plt.show()
