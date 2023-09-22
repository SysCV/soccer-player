import torch
import matplotlib.pyplot as plt


# Define the function for FL using your code
def compute_FL(x, kappa):
    smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

    FL = smoothing_cdf_start(torch.remainder(x, 1.0)) * (
        1 - smoothing_cdf_start(torch.remainder(x, 1.0) - 0.5)
    ) + smoothing_cdf_start(torch.remainder(x, 1.0) - 1) * (
        1 - smoothing_cdf_start(torch.remainder(x, 1.0) - 0.5 - 1)
    )

    return FL


# Create a range of x values from 0 to 1
x_values = torch.linspace(0, 1, 100)  # Adjust the number of points as needed

# Set the value of kappa
kappa = 0.2  # You can adjust this as needed

# Compute FL for each x value
FL_values = compute_FL(x_values, kappa)

# Plot the result
plt.plot(x_values, FL_values)
plt.xlabel("x")
plt.ylabel("FL")
plt.title("FL vs. x")
plt.grid(True)
plt.show()
