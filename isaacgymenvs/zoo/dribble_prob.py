import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math


class DistanceBasedSampler:
    def __init__(self, p_high, p_low, v_close, v_far):
        self.p_high = p_high
        self.p_low = p_low
        self.v_close = v_close
        self.v_far = v_far

    def sample(self, distance):
        p = self.smooth_prob(distance)
        return np.random.rand() < p

    def smooth_prob(self, distance):
        if distance < self.v_close:
            return self.p_high
        elif distance > self.v_far:
            return self.p_low
        else:
            # Linear interpolation
            return self.p_high + (self.p_low - self.p_high) * (
                distance - self.v_close
            ) / (self.v_far - self.v_close)


class RobotSimulation:
    def __init__(self):
        self.sampler = DistanceBasedSampler(0.08, 0.01, 0.1, 0.5)
        # Constants
        self.radius = 0.0  # Radius of the larger circular trajectory
        self.angular_speed = 0.0 * math.pi  # Angular speed in radians per second
        self.ball_vel = np.zeros(2)  # Linear speed of the ball in 2D
        self.ball_vel[0] = 1.0  # Linear speed of the ball in the x-direction
        self.ball_vel[1] = 0.0  # Linear speed of the ball in the y-direction

        # PD Control Constants (2D vectors)
        self.kp_pos = np.array(
            [300.0, 300.0]
        )  # Proportional gain for position control (x and y)
        self.kd_pos = np.array(
            [0.1, 0.1]
        )  # Derivative gain for position control (x and y)
        self.kp_ang = 0.0  # Proportional gain for angular speed control
        self.kd_ang = 0.0  # Derivative gain for angular speed control

        self.k1 = 0.8
        self.k2 = 4.0
        self.k3 = 0.5

        # Initial state of the rectangle (2D vector)
        self.rect_pos = np.array([-3.0, 0.0])
        self.rect_linear_speed = np.array(
            [1.0, 0.0]
        )  # Initial linear speed of the rectangle
        self.rect_angle = 0.0  # Initial angle of the rectangle (radians)
        self.rect_angular_speed = 0.0  # Initial angular speed of the rectangle

        # Rectangle Dimensions (1:2 aspect ratio)
        self.rect_length = 0.2
        self.rect_width = 0.2

        # Simulation Parameters
        self.dt = 0.02  # Time step (50 Hz)

        # Lists to store data for plotting
        self.ball_positions = []
        self.rect_positions = []
        self.rect_linear_speeds = []
        self.rect_angles = []
        self.rect_angular_speeds = []

        # Initialize previous values for the first frame
        self.prev_error_pos = np.zeros(2)
        self.prev_error_ang = 0.0
        self.prev_ball_pos = np.array([-3.0, 0.0])

        # Create a figure and axis for the plot
        self.fig, self.ax = plt.subplots()

        # Create an animation
        # self.ani = animation.FuncAnimation(
        #     self.fig, self.step, blit=False, interval=int(self.dt * 1000)
        # )

    def step(self, frame):
        if frame < 100:
            commands = np.array([1.0, 0.0])
        elif frame < 200:
            commands = np.array([-0.5, 0.0])
        else:
            commands = np.array([1.0, 1.0])

        # print("frame: ", frame, "commands: ", commands)

        if self.sampler.sample(np.linalg.norm(self.prev_ball_pos - self.rect_pos)):
            self.ball_vel = commands

        # Calculate the new position of the ball (2D vector)
        ball_pos = self.prev_ball_pos + np.array(
            [
                self.ball_vel[0] * self.dt,
                self.ball_vel[1] * self.dt,
            ]
        )

        # Calculate the error for position control (2D vector)
        error_pos = ball_pos - self.rect_pos

        # Calculate the error change (delta error)
        if frame > 0:
            delta_error_pos = error_pos - self.prev_error_pos
        else:
            delta_error_pos = np.zeros(2)  # Initialize with zeros for the first frame

        # Calculate the ball's velocity vector
        ball_speed = (ball_pos - self.prev_ball_pos) / self.dt

        delta_v = ball_speed - commands
        p_dog_target = ball_pos + delta_v * self.k1
        v_dog_target = (p_dog_target - self.rect_pos) * self.k2 + (
            ball_speed - self.rect_linear_speed
        ) * self.k3

        # Update the linear speed of the rectangle
        self.rect_linear_speed = v_dog_target

        # Calculate the desired angle to the ball
        target_angle = math.atan2(
            ball_pos[1] - self.rect_pos[1], ball_pos[0] - self.rect_pos[0]
        )

        # Calculate the error for angular speed control
        error_ang = target_angle - self.rect_angle

        # Ensure the error is within the range [-pi, pi]
        if error_ang > math.pi:
            error_ang -= 2 * math.pi
        elif error_ang < -math.pi:
            error_ang += 2 * math.pi

        # Calculate the change in error for angular speed control
        delta_error_ang = error_ang - self.prev_error_ang

        # Apply PD control to update the angular speed of the rectangle
        control_output_ang = self.kp_ang * error_ang - self.kd_ang * delta_error_ang

        # Update the angular speed of the rectangle
        self.rect_angular_speed = control_output_ang

        # Update the position and angle of the rectangle
        self.rect_pos[:2] += self.rect_linear_speed * self.dt
        self.rect_angle += self.rect_angular_speed * self.dt

        # Keep rect_angle within the range [-pi, pi]
        if self.rect_angle > math.pi:
            self.rect_angle -= 2 * math.pi
        elif self.rect_angle < -math.pi:
            self.rect_angle += 2 * math.pi

        # Store previous values for the next frame
        self.prev_error_pos = error_pos
        self.prev_error_ang = error_ang
        self.prev_ball_pos = ball_pos

        # Append data for plotting
        self.ball_positions.append(ball_pos)
        self.rect_positions.append(self.rect_pos)
        self.rect_linear_speeds.append(self.rect_linear_speed)
        self.rect_angles.append(self.rect_angle)
        self.rect_angular_speeds.append(self.rect_angular_speed)

    def show(self):
        # Clear the previous frame
        self.ax.clear()

        # Plot the ball at the new position
        self.ax.scatter(
            self.prev_ball_pos[0],
            self.prev_ball_pos[1],
            marker="o",
            color="blue",
            s=100,
        )

        # Plot the rotated rectangle (1:2 aspect ratio)
        rect_corner_x = self.rect_pos[0]  # X-coordinate of the rectangle's center
        rect_corner_y = self.rect_pos[1]  # Y-coordinate of the rectangle's center
        rect = plt.Rectangle(
            (
                rect_corner_x - self.rect_length / 2,
                rect_corner_y - self.rect_width / 2,
            ),
            self.rect_length,
            self.rect_width,
            angle=math.degrees(self.rect_angle),
            color="red",
        )
        self.ax.add_patch(rect)

        # Set axis limits
        self.ax.set_xlim(-5, 5)  # Adjust these limits as needed
        self.ax.set_ylim(-5, 5)


if __name__ == "__main__":
    # Create an instance of RobotSimulation
    robot_sim = RobotSimulation()

    # Define the number of frames for the simulation
    num_frames = 300  # Adjust as needed

    # Run the simulation and display the animation frame by frame
    for frame in range(num_frames):
        # Call the step method to calculate the next step of the simulation
        robot_sim.step(frame)

        # Call the show method to display the current frame
        robot_sim.show()

        # Pause for a short duration between frames to create the animation effect
        plt.pause(0.001)  # Adjust as needed

    # Close the plot window when the animation is done
    plt.show()
