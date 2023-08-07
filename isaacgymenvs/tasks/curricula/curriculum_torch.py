import torch
from matplotlib import pyplot as plt

def is_met(scale, l2_err, threshold):
    return (l2_err / scale) < threshold


def key_is_met(metric_cache, config, ep_len, target_key, env_id, threshold):
    # metric_cache[target_key][env_id] / ep_len
    scale = 1
    l2_err = 0
    return is_met(scale, l2_err, threshold)

class TorchCurriculum:
    def set_to(self, low, high, value=1.0):
        """
        Set the weights of the bins in the given range to the given value.
        :param low: The lower bound of the range.
        :param high: The upper bound of the range.
        :param value: The value to set the weights to.
        """
        inds = torch.logical_and(
            self.grid >= low[:, None],
            self.grid <= high[:, None]
        ).all(dim=0)

        assert len(inds) != 0, "You are initializing your distribution with an empty domain!"

        self.weights[inds] = value

    def __init__(self, device, **key_ranges):
        self.device = device

        self.cfg = cfg = {}
        self.indices = indices = {}
        for key, v_range in key_ranges.items():
            bin_size = (v_range[1] - v_range[0]) / v_range[2]
            cfg[key] = torch.linspace(v_range[0] + bin_size / 2, v_range[1] - bin_size / 2, v_range[2], device=device)  # centers

            indices[key] = torch.linspace(0, v_range[2] - 1, v_range[2], device=device)

        self.lows = torch.tensor([range[0] for range in key_ranges.values()], device=device)
        self.highs = torch.tensor([range[1] for range in key_ranges.values()], device=device)

        self.bin_sizes = {key: (v_range[1] - v_range[0]) / v_range[2] for key, v_range in key_ranges.items()}

        # torch's default meshgrid is ij
        self._raw_grid = torch.stack(torch.meshgrid(*cfg.values()))
        self._idx_grid = torch.stack(torch.meshgrid(*indices.values()))

        self.keys = [*key_ranges.keys()]

        self.grid = self._raw_grid.reshape([len(self.keys), -1])
        self.idx_grid = self._idx_grid.reshape([len(self.keys), -1])

        self._l = l = len(self.grid[0])
        self.ls = {key: len(self.cfg[key]) for key in self.cfg.keys()}

        self.weights = torch.zeros(l, device=device)

        self.weights_shaped = self.weights.view(*self.ls.values())
        self.indices = torch.arange(l, device=device)

    def __len__(self):
        return self._l

    def __getitem__(self, *keys):
        pass

    def update(self, **kwargs):
        # bump the envelope if
        pass

    def sample_bins(self, batch_size, low=None, high=None):
        """default to uniform"""
        if low is not None and high is not None:  # if bounds given
            valid_inds = torch.logical_and(
                self.grid >= low[:, None],
                self.grid <= high[:, None]
            ).all(dim=0)
            temp_weights = torch.zeros_like(self.weights, device=self.device)
            temp_weights[valid_inds] = self.weights[valid_inds]
            inds = torch.multinomial(temp_weights / temp_weights.sum(), batch_size, replacement=True)
        else:  # if no bounds given
            inds = torch.multinomial(self.weights / self.weights.sum(), batch_size, replacement=True)

        return self.grid.T[inds], inds

    def sample_uniform_from_cell(self, centroids):
        bin_sizes = torch.tensor([*self.bin_sizes.values()], device=self.device)
        low, high = centroids + bin_sizes / 2, centroids - bin_sizes / 2
        return torch.rand(len(centroids), device=self.device) * (high - low) + low

    def sample(self, batch_size, low=None, high=None):
        cgf_centroid, inds = self.sample_bins(batch_size, low=low, high=high)
        return torch.stack([self.sample_uniform_from_cell(v_range) for v_range in cgf_centroid]), inds


class RewardThresholdCurriculum(TorchCurriculum):
    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)

    def get_local_bins(self, bin_inds, ranges=0.1):
        if isinstance(ranges, float):
            ranges = torch.ones(self.grid.shape[0], device=self.device) * ranges

        # print("grid is", self.grid)
        
        bin_inds = bin_inds.reshape(-1)

        adjacent_inds = torch.logical_and(
            self.grid[:, None, :].repeat(1,bin_inds.shape[0], 1) >= self.grid[:, bin_inds, None] - ranges.reshape(-1, 1, 1),
            self.grid[:, None, :].repeat(1, bin_inds.shape[0], 1) <= self.grid[:, bin_inds, None] + ranges.reshape(-1, 1, 1)
        ).all(dim=0)

        return adjacent_inds

    def update(self, bin_inds, task_rewards, success_thresholds, local_range=0.5):
        is_success = 1.0
        for task_reward, success_threshold in zip(task_rewards, success_thresholds):
            is_success =  task_reward > success_threshold
        if len(success_thresholds) == 0:
            is_success = torch.tensor([False] * len(bin_inds), device=self.device)
        else:
            is_success = is_success

        self.weights[bin_inds[is_success]] = torch.clamp(self.weights[bin_inds[is_success]] + 0.2, 0, 1)
        # print("chosen bins: ", bin_inds[is_success])
        adjacents = self.get_local_bins(bin_inds[is_success], ranges=local_range)
        for adjacent in adjacents:
            adjacent_inds = adjacent.nonzero().squeeze()
            self.weights[adjacent_inds] = torch.clamp(self.weights[adjacent_inds] + 0.2, 0, 1)

if __name__ == '__main__':
    r = RewardThresholdCurriculum(device='cuda:0', x=(-1, 1, 5), y=(-1, 1, 6), z=(-1, 1, 11))

    assert r._raw_grid.shape == (3, 5, 6, 11), "grid shape is wrong: {}".format(r.grid.shape)  # the first dimension is (x, y, z)

    low, high = torch.tensor([-1, -1, -1], device='cuda:0'), torch.tensor([1, 1, 1], device='cuda:0')
    
    # r.set_to([-0.2, -0.6, -0.2], [0.2, 0.6, 0.2], value=1.0)
    
    r.set_to(low, high, value=1.0)

    samples, bins = r.sample(1000)

    plt.scatter(*samples.T[0:2].cpu())
    plt.show()

    plt.imshow(torch.mean(r.weights_shaped, dim=2).cpu(), cmap='gray',vmin=-1,vmax=1)
    plt.xticks([0,4],["-2","2"])
    plt.show()

    adjacents = r.get_local_bins(torch.tensor([10,]), ranges=0.3)
    # print(adjacents)
    for adjacent in adjacents:
        adjacent_inds = adjacent.nonzero().squeeze()
        # print(adjacent_inds)
        r.update(adjacent_inds,
                 [torch.ones(len(adjacent_inds)), torch.ones(len(adjacent_inds))],
                  [0.0, 0.0],
                  local_range=0.5)

    samples, bins = r.sample(1000)

    plt.scatter(*samples.T[0:2].cpu())
    plt.show()

    plt.imshow(torch.mean(r.weights_shaped, dim=2).cpu(), cmap='gray')
    plt.show()

    adjacents = r.get_local_bins(torch.tensor([10, ]), ranges=0.3)
    # print(adjacents)
    for adjacent in adjacents:
        adjacent_inds = adjacent.nonzero().squeeze()
        # print(adjacent_inds)
        r.update(adjacent_inds,
                 [torch.ones(len(adjacent_inds)), torch.ones(len(adjacent_inds))],
                  [0.0, 0.0],
                  local_range=0.5)

    samples, bins = r.sample(1000)

    plt.scatter(*samples.T[0:2].cpu())
    plt.show()

    plt.imshow(torch.mean(r.weights_shaped, dim=2).cpu(), cmap='gray')
    plt.show()
