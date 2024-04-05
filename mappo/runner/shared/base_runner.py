import torch

import numpy as np
import multiprocessing

from gym import spaces
from typing import Callable, Dict, List
from runner.shared.shared_buffer import SharedReplayBuffer
from utils.observation_space import MultiAgentObservationSpace

from algorithms.ramppo_network import R_MAPPO as TrainAlgo
from algorithms.rmappo_policy import R_MAPPOPolicy as Policy


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """

    def __init__(self, config):

        # config
        self.args = config["args"]
        self.train_env = config["train_env"]
        self.eval_env = config["eval_env"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]

        self.sigma = 1
        self.noise_dim = 10
        self.use_value_noise = True

        # name_parameter
        self.env_name: str = self.args.env_name
        self.algorithm_name: str = self.args.algorithm_name
        self.experiment_name: str = self.args.experiment_name

        # use_hyperparameter
        self.use_eval: bool = self.args.use_eval
        self.use_wandb: bool = self.args.use_wandb
        self.use_render: bool = self.args.use_render
        self.use_common_reward: bool = self.args.use_common_reward
        self.use_linear_lr_decay: bool = self.args.use_linear_lr_decay

        # parameters
        self.sampling_batch_size: int = self.args.sampling_batch_size
        self.hidden_size: int = self.args.hidden_size
        self.recurrent_N: int = self.args.recurrent_N
        self.episode_length: int = self.args.max_step
        self.max_episodes: int = self.args.max_episodes
        self.eval_batch_size = self.args.eval_batch_size

        # interval
        self.log_interval: int = self.args.log_interval
        self.save_interval: int = self.args.save_interval
        self.eval_interval: int = self.args.eval_interval

        # render time
        self.sleep_second: float = self.args.sleep_second

        # hardware_settings
        # self.queue = multiprocessing.Queue()

        # share_observation
        self.observation_space = self.train_env[0].observation_space

        self._obs_high = np.tile(self.train_env[0]._obs_high, self.num_agents)
        self._obs_low = np.tile(self.train_env[0]._obs_low, self.num_agents)
        self.share_observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.num_agents)]
        )

        process_reward: Dict[bool, object] = {
            True: self.convert_sum_rewards,
            False: self.convert_each_rewards,
        }
        self.process_reward_type: Callable[[bool], object] = process_reward.get(self.use_common_reward)

        # policy network
        self.policy = Policy(
            args=self.args,
            obs_space=self.observation_space[0],
            cent_obs_space=self.share_observation_space[0],
            act_space=self.train_env[0].action_space[0],
            device=self.device,
        )

        # algorithm
        self.trainer = TrainAlgo(args=self.args, policy=self.policy, device=self.device)
        # buffer
        self.buffer = SharedReplayBuffer(
            args=self.args,
            num_agents=self.num_agents,
            obs_space=self.observation_space[0],
            cent_obs_space=self.share_observation_space[0],
            act_space=self.train_env[0].action_space[0],
        )

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """
        Calculate the expected value of the action performed by the agent, and use these expected values to calculate the return value

        :param next_values: Value function predictions calculated by Critic(advantages? or v values?)
        :param value_normalizer: Normalization function/class that depends on the use_propart, use_valuenorm hyperparameter value
        """
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.sampling_batch_size))
        self.buffer.compute_returns(next_value=next_values, value_normalizer=self.trainer.value_normalizer)

    def train(self, noise_vector=None):
        """Train policies with data in buffer."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(buffer=self.buffer, noise_vector=noise_vector)
        self.buffer.after_update()
        return train_infos

    def obs_sharing(self, obs: List, warm_up=False) -> np.array:
        if warm_up:
            share_obs = np.array(obs).reshape(-1)
            share_obs_list = np.array([share_obs for _ in range(self.num_agents)])
        else:
            share_obs_list = np.array(
                [[np.array(each_obs).reshape(-1) for _ in range(self.num_agents)] for each_obs in obs]
            )
        return share_obs_list

    def convert_each_rewards(self, rewards_batch):
        converted_rewards = [[[reward] for reward in rewards] for rewards in rewards_batch]
        return converted_rewards

    def convert_sum_rewards(self, rewards_batch):
        converted_rewards = [[[sum(rewards)] for _ in range(self.num_agents)] for rewards in rewards_batch]
        return converted_rewards
