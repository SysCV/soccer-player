import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt

# Load the WAV file
file_path = "/home/gymuser/IsaacGymEnvs-main/assets/audio/out.wav"
sample_rate, audio_data = wavfile.read(file_path)


# Read audio data and convert it to a NumPy array
audio_data = audio_data.astype(np.float32) / 32768.0  # Scale to the range [-1, 1]

plt.figure(figsize=(10, 4))
plt.plot(audio_data, color="b")
plt.title("Audio Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Calculate the number of samples in a 5-second range
total_samples = 1 * sample_rate

window_per_seconds = 50
# Calculate the number of samples per 1/50 second window (20 ms)
window_size = sample_rate // window_per_seconds  # 1/50 second window

# Calculate the number of non-overlapping windows
num_windows = total_samples // window_size

# Initialize arrays to store RMS values and time values
rms_values = []
time_values = []

start_time_slide = 3

# Iterate through the audio data in 1/50 second windows
for i in range(
    start_time_slide * num_windows * window_size,
    (start_time_slide + 2) * num_windows * window_size,
    window_size,
):
    window = audio_data[i : i + window_size]
    rms = np.sqrt(np.mean(np.square(window)))
    rms_values.append(rms)
    time_values.append(i / sample_rate)  # Convert sample index to time in seconds
joint_positions = np.load(
    "/home/gymuser/IsaacGymEnvs-main/assets/audio/joint_positions.npy"
)

fig, axs = plt.subplots(2, 1, figsize=(12, 5))


axs[0].plot(
    time_values,
    joint_positions[start_time_slide * 50 : (start_time_slide + 2) * 50, :],
    linestyle="-",
    label="Measured",
)
# np.save("joint_positions.npy", joint_positions)
axs[0].set_title("Joint Positions")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Joint Position (rad)")

axs[1].plot(time_values, rms_values, color="b")
axs[1].set_title("RMS Over Time (1/{} second windows)".format(window_per_seconds))
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("RMS")

plt.tight_layout()
plt.show()
