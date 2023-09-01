from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.tr_helpers import unsqueeze_obs
import gym
import torch
from torch import nn
import numpy as np
import time


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


class Player(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config["network"]
        self.actions_num = self.action_space.shape[0]
        self.actions_low = (
            torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        )
        self.actions_high = (
            torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        )
        self.mask = [False]

        self.normalize_input = self.config["normalize_input"]
        self.normalize_value = self.config.get("normalize_value", False)

        obs_shape = self.obs_shape
        config = {
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            "num_seqs": self.num_agents,
            "value_size": self.env_info.get("value_size", 1),
            "normalize_value": self.normalize_value,
            "normalize_input": self.normalize_input,
        }
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

        self.generate_offline_data = self.player_config.get(
            "generate_offline_data", False
        )

        if self.generate_offline_data:
            print("preparing offline dataset...")
            self.offline_data_length = self.player_config.get("offline_data_length")
            self.offline_save_dir = self.player_config.get("offline_save_dir")
            self.discount_factor = self.player_config.get("discount_factor")
            self.prepare_offline_dataset()

    def prepare_offline_dataset(self):
        self.max_steps = self.offline_data_length - 1
        self.offline_dataset = VectorizedReplayBuffer(
            self.env.num_envs,
            self.observation_space,
            self.action_space,
            self.offline_data_length,
            self.device,
        )

    def get_action(self, obs, is_deterministic=False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.states,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict["mus"]
        action = res_dict["actions"]
        self.states = res_dict["rnn_states"]
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(
                self.actions_low,
                self.actions_high,
                torch.clamp(current_action, -1.0, 1.0),
            )
        else:
            return current_action

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        if self.normalize_input and "running_mean_std" in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

        env_state = checkpoint.get("env_state", None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def reset(self):
        self.init_rnn()

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()
        vec_env_steps = 1
        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)

            if self.generate_offline_data:
                self.write_first_step(obses)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                obses, r, done, info = self.env_step(self.env, action)
                if self.generate_offline_data:
                    self.write_step(obses, action, r, done, info)
                cr += r
                steps += 1
                vec_env_steps += 1
                if vec_env_steps % 5 == 0:
                    print("vec_env_steps", vec_env_steps)

                if render:
                    self.env.render(mode="human")
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[:: self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if "battle_won" in info:
                            print_game_res = True
                            game_res = info.get("battle_won", 0.5)
                        if "scores" in info:
                            print_game_res = True
                            game_res = info.get("scores", 0.5)

                    if self.print_stats:
                        cur_rewards_done = cur_rewards / done_count
                        cur_steps_done = cur_steps / done_count
                        if print_game_res:
                            print(
                                f"reward: {cur_rewards_done:.1f} steps: {cur_steps_done:.1} w: {game_res:.1}"
                            )
                        else:
                            print(
                                f"reward: {cur_rewards_done:.1f} steps: {cur_steps_done:.1f}"
                            )

                    sum_game_res += game_res
                    if not (self.generate_offline_data) and (
                        self.batch_size // self.num_agents == 1
                        or games_played >= n_games
                    ):
                        break

                if (
                    self.generate_offline_data
                    and vec_env_steps >= self.offline_data_length
                ):
                    games_played = n_games + 1  # force the loop break
                    print("data collection finished")
                    break

        if self.generate_offline_data:
            assert self.offline_dataset.idx == self.offline_data_length
            self.offline_dataset.save_tensor_dict(self.offline_save_dir)

        print(sum_rewards)
        if print_game_res:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
                "winrate:",
                sum_game_res / games_played * n_game_life,
            )
        else:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
            )

    def write_first_step(self, obs):
        dataset_dict = {}
        dataset_dict["state_obs"] = obs["state_obs"]
        dataset_dict["is_terminal"] = torch.zeros(
            self.env.num_envs, dtype=torch.int32, device=self.device
        )
        dataset_dict["is_first"] = torch.ones(
            self.env.num_envs, dtype=torch.int32, device=self.device
        )  # First Step is always after reset
        dataset_dict["reward"] = torch.zeros(
            self.env.num_envs, dtype=torch.float32, device=self.device
        )
        dataset_dict["discount"] = (
            torch.ones(self.env.num_envs, dtype=torch.float32, device=self.device)
            * self.discount_factor
        )
        dataset_dict["action"] = torch.zeros(
            self.env.num_envs,
            self.env.action_space.shape[0],
            dtype=torch.float32,
            device=self.device,
        )
        dataset_dict["logprob"] = torch.zeros(
            self.env.num_envs, dtype=torch.float32, device=self.device
        )
        self.offline_dataset.add(dataset_dict)

    def write_step(self, obs, action, reward, done, info):
        dataset_dict = {}
        is_terminal = done.to(self.device) & ~info["time_outs"].to(self.device)
        # discount_factor = is_terminal * 0.997
        discount_factor = (~is_terminal) * self.discount_factor

        # Fill obs dict with other return
        dataset_dict["state_obs"] = obs["state_obs"]
        dataset_dict["is_terminal"] = is_terminal.int()
        dataset_dict["is_first"] = info["is_firsts"].int().to(self.device)
        dataset_dict["reward"] = reward
        dataset_dict["discount"] = discount_factor
        dataset_dict["action"] = action
        dataset_dict["logprob"] = torch.zeros(
            self.env.num_envs, dtype=torch.float32, device=self.device
        )

        self.offline_dataset.add(dataset_dict)


class VectorizedReplayBuffer:
    def __init__(
        self, num_envs, obs_shape, act_shape, capacity, device, offline_data_path=None
    ):
        """Create Vectorized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        See Also
        --------
        ReplayBuffer.__init__
        """

        self.device = device

        self.tensor_dict = {}
        if offline_data_path is not None:
            self.load_tensor_dict(offline_data_path)
            self.width = self.tensor_dict["state_obs"].shape[0]
            assert self.width == num_envs, "num_envs not match"
            self.length = self.tensor_dict["state_obs"].shape[1]
            assert self.length == capacity, "capacity not match"
            self.idx = self.capacity
            self.full = True
            return
        else:
            self.width = num_envs
            self.length = capacity
            base_shape = (num_envs, capacity)

            # init obs with obs_shape
            self.tensor_dict = self._create_tensor_from_space(
                obs_shape, base_shape
            )  # must be dict

            self.tensor_dict["action"] = self._create_tensor_from_space(
                act_shape, base_shape
            )  # must be dict
            self.tensor_dict["logprob"] = torch.empty(
                (
                    num_envs,
                    capacity,
                ),
                dtype=torch.int32,
                device=device,
            )  # must be dict

            # add common keys
            self.tensor_dict["is_terminal"] = torch.empty(
                (
                    num_envs,
                    capacity,
                ),
                dtype=torch.int32,
                device=device,
            )
            self.tensor_dict["is_first"] = torch.empty(
                (
                    num_envs,
                    capacity,
                ),
                dtype=torch.int32,
                device=device,
            )
            self.tensor_dict["reward"] = torch.empty(
                (
                    num_envs,
                    capacity,
                ),
                dtype=torch.float32,
                device=device,
            )
            self.tensor_dict["discount"] = torch.empty(
                (
                    num_envs,
                    capacity,
                ),
                dtype=torch.float32,
                device=device,
            )

            self.capacity = capacity
            self.idx = 0  # pointer
            self.full = False

    def add(self, transition_dict):
        for key in transition_dict.keys():
            assert key in self.tensor_dict.keys(), f"key {key} not in tensor_dict"

        for k, v in transition_dict.items():
            # print(k,v)
            # print(self.tensor_dict[k][:,self.idx])
            self.tensor_dict[k][:, self.idx] = v

        self.idx = self.idx + 1
        assert self.idx <= self.capacity, "idx is larger than capacity"

        # self.tensor_dict["is_first"][:, self.idx] = 1
        self.full = self.full or self.idx == 0

    def sample(self, batch_length):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_length: int
            How many transitions to sample.
        Returns
        -------
        obses: torch tensor
            batch of observations
        actions: torch tensor
            batch of actions executed given obs
        rewards: torch tensor
            rewards received as results of executing act_batch
        next_obses: torch tensor
            next set of observations seen after executing act_batch
        not_dones: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not
        not_dones_no_max: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not, specifically exlcuding maximum episode steps
        """

        start_id = torch.randint(
            0, self.capacity if self.full else self.idx, (1,), device=self.device
        ).item()

        tail = self.capacity if self.full else self.idx
        remaining_capacity = tail - start_id

        batch_dict = {}
        if remaining_capacity < batch_length:  # need concate
            for k, v in self.tensor_dict.items():
                batch_dict[k] = torch.cat(
                    (v[:, start_id:tail], v[:, : batch_length - remaining_capacity]),
                    dim=1,
                )
        else:
            for k, v in self.tensor_dict.items():
                batch_dict[k] = v[:, start_id : start_id + batch_length]

        return batch_dict

    def _create_tensor_from_space(self, space, base_shape):
        if type(space) is gym.spaces.Box:
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            print("creating tensor with shape", base_shape + space.shape)
            print("dtype is", dtype)
            return torch.zeros(
                base_shape + space.shape, dtype=dtype, device=self.device
            )
        # if type(space) is gym.spaces.Discrete:
        #     dtype = numpy_to_torch_dtype_dict[space.dtype]
        #     return torch.zeros(base_shape, dtype= dtype, device = self.device)
        # if type(space) is gym.spaces.Tuple:
        #     '''
        #     assuming that tuple is only Discrete tuple
        #     '''
        #     dtype = numpy_to_torch_dtype_dict[space.dtype]
        #     tuple_len = len(space)
        #     return torch.zeros(base_shape +(tuple_len,), dtype= dtype, device = self.device)
        elif type(space) is gym.spaces.Dict:
            t_dict = {}
            for k, v in space.spaces.items():
                print("now allocating space for key", k)
                t_dict[k] = self._create_tensor_from_space(v, base_shape)
            return t_dict
        else:
            raise NotImplementedError("space type not implemented!!")

    def save_tensor_dict(self, fn):
        torch.save(self.tensor_dict, fn)

    def load_tensor_dict(self, fn):
        self.tensor_dict = torch.load(fn)


numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float64"): torch.float32,
    np.dtype("float32"): torch.float32,
    # np.dtype('float64')    : torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}
