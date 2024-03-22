   
import os
import time
import wandb
import torch

import numpy as np

from gym import spaces
from replay_buffer.separated_buffer import SeparatedReplayBuffer
from runner.shared.observation_space import MultiAgentObservationSpace


def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.args = config['args']
        self.envs = config['train_env']
        self.eval_envs = config['eval_env']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # name_parameter
        self.env_name: str = self.args.env_name
        self.algorithm_name: str = self.args.algorithm_name
        self.experiment_name: str = self.args.experiment_name

        # rollout_threads
        self.n_rollout_threads: int = self.args.n_rollout_threads
        self.n_eval_rollout_threads: int = self.args.n_eval_rollout_threads
        self.n_render_rollout_threads: int = self.args.n_render_rollout_threads

        # use_hyperparameter
        self.use_eval: bool = self.args.use_eval
        self.use_wandb: bool = self.args.use_wandb
        self.use_render: bool = self.args.use_render
        self.use_centralized_V: bool = self.args.use_centralized_V
        self.use_linear_lr_decay: bool = self.args.use_linear_lr_decay

        # parameters
        self.max_episodes: int = self.args.max_episodes
        self.episode_length: int = self.args.max_step
        self.hidden_size: int = self.args.hidden_size
        self.recurrent_N: int = self.args.recurrent_N
        self.batch_size: int = self.args.batch_size  
        self.eval_episodes: int = self.args.eval_episodes

        # interval
        self.log_interval: int = self.args.log_interval
        self.save_interval: int = self.args.save_interval
        self.eval_interval: int = self.args.eval_interval

        # render time
        self.sleep_second: float = self.args.sleep_second



        from algorithms.ramppo_network import R_MAPPO as TrainAlgo
        from algorithms.policys.rmappo_policy import R_MAPPOPolicy as Policy


        self.policy = []
        for agent_id in range(self.num_agents):
            self.observation_space = self.train_env.observation_space
            if self.use_centralized_V:
                self._obs_high = np.tile(self.train_env._obs_high, self.num_agents)
                self._obs_low = np.tile(self.train_env._obs_low, self.num_agents)
                self.share_observation_space = MultiAgentObservationSpace(
                    [
                        spaces.Box(self._obs_low, self._obs_high)
                        for _ in range(self.num_agents)
                    ]
                )
            # policy network
            po = Policy(self.args,
                        self.envs.observation_space[agent_id],
                        share_observation_space[agent_id],
                        self.envs.action_space[agent_id],
                        device = self.device)
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.args, self.policy[agent_id], device = self.device)
            # buffer
            share_observation_space = self.envs.observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)
            
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
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)       
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom = self.trainer[agent_id].value_normalizer
                torch.save(policy_vnrom.state_dict(), str(self.save_dir) + "/vnrom_agent" + str(agent_id) + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom_state_dict = torch.load(str(self.model_dir) + '/vnrom_agent' + str(agent_id) + '.pt')
                self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
