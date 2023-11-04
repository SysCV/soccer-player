import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    test_distances = [0.1, 0.2, 0.3, 0.4, 0.5]

    sampler = DistanceBasedSampler(0.02, 0.006, 0.1, 0.5)
    # Test the class
    test_results_class = {
        d: sum([sampler.sample(d) for _ in range(500000)]) for d in test_distances
    }
    print(test_results_class)

    distances = np.linspace(0, 1, 400)
    probs = [sampler.smooth_prob(d) for d in distances]

    plt.plot(distances, probs)
    plt.xlabel("Distance")
    plt.ylabel("Probability")
    plt.title("Probability vs Distance using Modified Sigmoid")
    plt.grid(True)
    plt.show()
