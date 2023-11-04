import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

import numpy as np

input_sequence = np.loadtxt("./dataset/input_sequence.csv")
output_sequence = np.loadtxt("./dataset/output.csv")

# plot the input sequence
plt.plot(input_sequence[:], "r")
plt.plot(output_sequence[:], "b")
plt.show()
