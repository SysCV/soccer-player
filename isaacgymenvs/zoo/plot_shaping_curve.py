import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("./shaping.csv")


# Sampling frequency (e.g., every 10th data point)
sampling_freq = 100
sampled_df = df.iloc[::sampling_freq, :]

# Assuming the first column is the x-axis and the rest are y-values
x_values = sampled_df.iloc[:, 0]
y_values = sampled_df.iloc[:, 1:]


# Exponential Smoothing Function
def exponential_smoothing(series, alpha):
    result = [series.iloc[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series.iloc[n] + (1 - alpha) * result[n - 1])
    return pd.Series(result, index=series.index)


# Apply exponential smoothing
alpha = 0.7  # Smoothing level
smoothed_data = y_values.apply(lambda x: exponential_smoothing(x, alpha))

# Calculate mean and standard deviation of the smoothed data
mean_smoothed = smoothed_data.mean(axis=1)
std_smoothed = smoothed_data.std(axis=1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, mean_smoothed, label="Smoothed Mean Curve", color="black")
plt.fill_between(
    x_values,
    mean_smoothed - 1 * std_smoothed,
    mean_smoothed + 1 * std_smoothed,
    color="gray",
    alpha=0.2,
    label="Smoothed Standard Deviation",
)
plt.title("Smoothed Mean and Standard Deviation of Curves")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
