import numpy as np
import random
import gym
import torch
from rl_games.common.segment_tree import SumSegmentTree, MinSegmentTree
import torch

from rl_games.algos_torch.torch_ext import numpy_to_torch_dtype_dict
from rl_games.common.experience import ExperienceBuffer

class FuExperienceBuffer(ExperienceBuffer):
    def _init_from_env_info(self, env_info):
        super()._init_from_env_info(env_info)
        obs_base_shape = self.obs_base_shape
        self.tensor_dict['obses_fu'] = self._create_tensor_from_space(env_info['observation_space'], obs_base_shape)
