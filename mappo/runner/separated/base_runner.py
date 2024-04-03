import wandb
import torch

import numpy as np

from gym import spaces
from typing import Callable, Dict, List
from runner.separated.separated_buffer import SeparatedReplayBuffer
from utils.observation_space import MultiAgentObservationSpace

from algorithms.ramppo_network import R_MAPPO as TrainAlgo
from algorithms.rmappo_policy import R_MAPPOPolicy as Policy

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.args = config['args']
        self.train_env = config['train_env']
        self.eval_env = config['eval_env']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # name_parameter
        self.env_name: str = self.args.env_name
        self.algorithm_name: str = self.args.algorithm_name
        self.experiment_name: str = self.args.experiment_name

        # batch setting
        self.sampling_batch_size: int = self.args.sampling_batch_size
        self.eval_rollout_threads = 1

        # use_hyperparameter
        self.use_eval: bool = self.args.use_eval
        self.use_wandb: bool = self.args.use_wandb
        self.use_render: bool = self.args.use_render
        self.use_common_reward: bool = self.args.use_common_reward
        self.use_centralized_V: bool = self.args.use_centralized_V
        self.use_linear_lr_decay: bool = self.args.use_linear_lr_decay

        # parameters
        self.hidden_size: int = self.args.hidden_size
        self.recurrent_N: int = self.args.recurrent_N
        self.max_episodes: int = self.args.max_episodes
        self.episode_length: int = self.args.max_step

        # interval
        self.log_interval: int = self.args.log_interval
        self.save_interval: int = self.args.save_interval
        self.eval_interval: int = self.args.eval_interval

        # render time
        self.sleep_second: float = self.args.sleep_second

        self.observation_space = self.train_env[0].observation_space
        
        if self.use_centralized_V:
            self._obs_high = np.tile(self.train_env[0]._obs_high, self.num_agents)
            self._obs_low = np.tile(self.train_env[0]._obs_low, self.num_agents)
            self.share_observation_space = MultiAgentObservationSpace(
                [
                    spaces.Box(self._obs_low, self._obs_high)
                    for _ in range(self.num_agents)
                ]
            )
        else:
            self.share_observation_space = self.observation_space
        print(self.share_observation_space)
        process_obs: Dict[bool, object] = {True : self.obs_sharing, False: self.obs_isolated}
        self.process_obs_type: Callable[[bool], object] = process_obs.get(self.use_centralized_V)

        process_reward: Dict[bool, object] = {True : self.convert_sum_rewards, False: self.convert_each_rewards}
        self.process_reward_type: Callable[[bool], object] = process_reward.get(self.use_common_reward)

        self.policy: List[object] = []
        self.trainer: List[object] = []
        self.buffer: List[object] = []

        for agent_id in range(self.num_agents): 
            po = Policy(self.args,
                        self.train_env[0].observation_space[agent_id],
                        self.share_observation_space[agent_id],
                        self.train_env[0].action_space[agent_id],
                        device = self.device)
            
            self.policy.append(po)

            tr = TrainAlgo(self.args, self.policy[agent_id], device = self.device)
            
            bu = SeparatedReplayBuffer(self.args,
                                       self.train_env[0].observation_space[agent_id],
                                       self.share_observation_space[agent_id],
                                       self.train_env[0].action_space[agent_id])
            
            self.trainer.append(tr)
            self.buffer.append(bu)



    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(buffer = self.buffer[agent_id])
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()
        return train_infos

    def obs_sharing(self, obs: List, warm_up = False) -> np.array:
        if warm_up:
            share_obs = np.array(obs).reshape(-1)
            share_obs_list = np.array([share_obs for _ in range(self.num_agents)]) 
        else: 
            share_obs_list = np.array([[np.array(each_obs).reshape(-1) for _ in range(self.num_agents)] for each_obs in obs])
        return share_obs_list
    
    def obs_isolated(self, obs: List, warm_up = False) -> np.array:
        if warm_up:
            isolated_obs_list = np.array(obs)
        else:
            isolated_obs_list = np.array(obs)
        return isolated_obs_list
    
    def convert_each_rewards(self, rewards_batch):
        converted_rewards = [[[reward] for reward in rewards] for rewards in rewards_batch]
        return converted_rewards

    def convert_sum_rewards(self, rewards_batch):
        converted_rewards = [[[sum(rewards)] for _ in range(self.num_agents)] for rewards in rewards_batch]
        return converted_rewards

    
    def log_train(self, train_infos, eval_result):

        total_train_infos = {key: 0.0 for key in train_infos[0]}
        total_train_infos["Test_Rewards"] = eval_result

        for agent_i in range(self.num_agents):
            for key in train_infos[agent_i].keys():
                total_train_infos[key] += train_infos[agent_i][key]

        total_train_infos["ratio"] /= self.num_agents
        total_train_infos["value_loss"] /= self.num_agents
        total_train_infos["dist_entropy"] /= self.num_agents
        total_train_infos["actor_grad_norm"] /= self.num_agents
        total_train_infos["critic_grad_norm"] /= self.num_agents

        if self.use_wandb:
            wandb.log(total_train_infos)
