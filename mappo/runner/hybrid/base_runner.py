import wandb
import torch

import numpy as np

from gym import spaces
from typing import Callable, Dict, List
from runner.separated.separated_buffer import SeparatedReplayBuffer
from utils.observation_space import MultiAgentObservationSpace

from utils.util import obs_sharing, obs_isolated, convert_each_rewards, convert_sum_rewards

from runner.hybrid.algorithms.ramppo_network import R_MAPPO as TrainAlgo
from runner.hybrid.algorithms.policy.rmappo_actor_policy import R_MAPPO_Actor_Policy as Actor_Policy
from runner.hybrid.algorithms.policy.rmappo_critic_policy import R_MAPPO_Critic_Policy as Critic_Policy
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
        self.eval_interval: int = self.args.eval_interval
        self.render_interval: int = self.args.render_interval

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
        
        process_obs: Dict[bool, object] = {True : obs_sharing, False: obs_isolated}
        self.process_obs_type: Callable[[bool], object] = process_obs.get(self.use_centralized_V)

        process_reward: Dict[bool, object] = {True : convert_sum_rewards, False: convert_each_rewards}
        self.process_reward_type: Callable[[bool], object] = process_reward.get(self.use_common_reward)

        self.actor_policy: List[object] = []
        for agent_id in range(self.num_agents): 
            actor_policy = Actor_Policy(
                args = self.args,
                obs_space = self.train_env[0].observation_space[agent_id],
                act_space = self.train_env[0].action_space[agent_id],
                device = self.device
            )
            self.actor_policy.append(actor_policy)
        self.critic_policy = Critic_Policy(
            args = self.args,
            obs_space = self.train_env[0].observation_space[agent_id],
            cent_obs_space = self.share_observation_space[agent_id],
            act_space = self.train_env[0].action_space[agent_id],
            device = self.device
        )

        self.trainer: List[object] = []
        self.buffer: List[object] = []
        for agent_id in range(self.num_agents): 
            tr = TrainAlgo(
                args = self.args, 
                actor_policy = self.actor_policy[agent_id],
                critic_policy = self.critic_policy, 
                device = self.device
            )
            
            bu = SeparatedReplayBuffer(
                self.args,
                self.train_env[0].observation_space[agent_id],
                self.share_observation_space[agent_id],
                self.train_env[0].action_space[agent_id]
            )
            
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
            next_value = self.trainer[agent_id].critic_policy.get_values(
                self.buffer[agent_id].share_obs[-1], 
                self.buffer[agent_id].rnn_states_critic[-1],
                self.buffer[agent_id].masks[-1]
            )
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(
                central_buffer = self.buffer,
                agent_id = agent_id,
                update_actor = True
            )
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()
        return train_infos

    def log_train(self, train_infos, eval_result):

        total_train_infos = {key: 0.0 for key in train_infos[0]}
        total_train_infos["Test_Rewards"] = eval_result

        for agent_i in range(self.num_agents):
            for key in train_infos[agent_i].keys():
                total_train_infos[key] += train_infos[agent_i][key]

        total_train_infos["ratio"] = total_train_infos["ratio"]/self.num_agents

        total_train_infos["dist_entropy"] /= self.num_agents
        total_train_infos["actor_grad_norm"] /= self.num_agents

        if self.use_wandb:
            wandb.log(total_train_infos)

    def train_log(self):
        return 
