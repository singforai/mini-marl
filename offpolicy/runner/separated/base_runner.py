import os
import numpy as np
import wandb
from tqdm import tqdm 
import torch
import time
import datetime
from gym import spaces
from typing import List, Dict, Callable
from utils.observation_space import MultiAgentObservationSpace
from utils.rec_buffer import RecReplayBuffer, PrioritizedRecReplayBuffer
from utils.util import DecayThenFlatSchedule, get_cent_act_dim, get_dim_from_space

class RecRunner(object):
    """Base class for training recurrent policies."""

    def __init__(self, config):
        """
        Base class for training recurrent policies.
        :param config: (dict) Config dictionary containing parameters for training.
        """
        self.args = config["args"]
        self.train_env = config["train_env"]
        self.eval_env = config["eval_env"]
        self.num_agents = config["num_agents"]
        self.device = config["device"]

        self.q_learning: List[str] = ["qmix", "vdn", "mqmix", "mvdn"]
        self.agent_ids: List[int] = [i for i in range(self.num_agents)]

        self.use_avail_acts: bool = False
        self.use_per: bool = self.args.use_per
        self.use_eval: bool = self.args.use_eval
        self.use_wandb: bool = self.args.use_wandb
        self.use_popart: bool = self.args.use_popart
        self.share_policy: bool = self.args.share_policy
        self.use_rnn_layer: bool = self.args.use_rnn_layer
        self.use_soft_update: bool = self.args.use_soft_update
        self.use_same_share_obs: bool = self.args.use_same_share_obs
        self.use_reward_normalization: bool = self.args.use_reward_normalization

        self.env_name: str = self.args.env_name
        self.algorithm_name: str = self.args.algorithm_name
        self.experiment_name: str = self.args.experiment_name

        # no parallel envs
        self.num_envs: int = 1
        self.batch_size: int = self.args.batch_size
        self.hidden_size: int = self.args.hidden_size
        self.buffer_size: int = self.args.buffer_size
        self.log_interval: int = self.args.log_interval
        self.max_episodes: int = self.args.max_episodes
        self.save_interval: int = self.args.save_interval
        self.eval_interval: int = self.args.eval_interval
        self.num_env_steps: int = self.args.num_env_steps
        self.train_interval: int = self.args.train_interval
        self.episode_length: int = self.args.episode_length
        self.train_interval_episode: int = self.args.train_interval_episode 
        self.actor_train_interval_step: int = self.args.actor_train_interval_step
        self.popart_update_interval_step: int = self.args.popart_update_interval_step
        self.hard_update_interval_episode: int = self.args.hard_update_interval_episode
        
        self.per_alpha: float = self.args.per_alpha
        self.per_beta_start: float = self.args.per_beta_start

        self.observation_space = self.train_env.observation_space
        self._obs_high: List[float] = np.tile(self.train_env._obs_high, self.num_agents)
        self._obs_low: List[float] = np.tile(self.train_env._obs_low, self.num_agents)
        self.share_observation_space = MultiAgentObservationSpace(
            [
                spaces.Box(self._obs_low, self._obs_high)
                for _ in range(self.num_agents)
            ]
        )

        process_reward: Dict[bool, object] = {True : self.convert_sum_rewards, False: self.convert_each_rewards}
        self.process_reward_type: Callable[[bool], object] = process_reward.get(self.args.use_common_reward)

        self.policy_info = {
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(self.share_observation_space[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(self.train_env.action_space),
                                        "obs_space": self.train_env.observation_space[agent_id],
                                        "share_obs_space": self.share_observation_space[agent_id],
                                        "act_space": self.train_env.action_space[agent_id]}
            for agent_id in range(self.num_agents)
        }

        def policy_mapping_fn(agent_id): return 'policy_' + str(agent_id)

        self.policy_ids = sorted(list(self.policy_info.keys()))

        self.total_env_steps = 0  
        self.num_episodes_collected = 0
        self.total_train_steps = 0  
        self.last_train_episode = 0  
        self.last_eval_T = 0  
        self.last_save_T = 0  
        self.last_log_T = 0 
        self.last_hard_update_episode = 0 

        if self.args.use_naive_recurrent_policy:
            self.data_chunk_length: int = self.episode_length 
        else:
            self.data_chunk_length: int = self.args.data_chunk_length

        if self.use_rnn_layer:
            if self.algorithm_name == "rmatd3":
                from algorithms.r_matd3.algorithm.rMATD3Policy import R_MATD3Policy as Policy
                from algorithms.r_matd3.r_matd3 import R_MATD3 as TrainAlgo
            elif self.algorithm_name == "rmaddpg":
                assert self.actor_train_interval_step == 1, (
                    "rmaddpg only supports actor_train_interval_step=1.")
                from algorithms.r_maddpg.algorithm.rMADDPGPolicy import R_MADDPGPolicy as Policy
                from algorithms.r_maddpg.r_maddpg import R_MADDPG as TrainAlgo
            elif self.algorithm_name == "qmix":
                from algorithms.qmix.algorithm.QMixPolicy import QMixPolicy as Policy
                from algorithms.qmix.qmix import QMix as TrainAlgo
            elif self.algorithm_name == "vdn":
                from algorithms.vdn.algorithm.VDNPolicy import VDNPolicy as Policy
                from algorithms.vdn.vdn import VDN as TrainAlgo
            else:
                raise NotImplementedError
        else:
            if self.algorithm_name == "matd3":
                from algorithms.matd3.algorithm.MATD3Policy import MATD3Policy as Policy
                from algorithms.matd3.matd3 import MATD3 as TrainAlgo
            elif self.algorithm_name == "maddpg":
                from algorithms.maddpg.algorithm.MADDPGPolicy import MADDPGPolicy as Policy
                from algorithms.maddpg.maddpg import MADDPG as TrainAlgo
            elif self.algorithm_name == "mqmix":
                from algorithms.mqmix.algorithm.mQMixPolicy import M_QMixPolicy as Policy
                from algorithms.mqmix.mqmix import M_QMix as TrainAlgo
            elif self.algorithm_name == "mvdn":
                from algorithms.mvdn.algorithm.mVDNPolicy import M_VDNPolicy as Policy
                from algorithms.mvdn.mvdn import M_VDN as TrainAlgo
            else:
                raise NotImplementedError
                
        self.train = self.batch_train_q if self.algorithm_name in self.q_learning else self.batch_train

        self.policies = {p_id: Policy(
            args = self.args, 
            config = config, 
            policy_config = self.policy_info[p_id]
        ) for p_id in self.policy_ids}

        # initialize trainer class for updating policies
        self.trainer = TrainAlgo(
            args = self.args, 
            num_agents = self.num_agents,
            batch_size = self.batch_size, 
            policies = self.policies, 
            policy_mapping_fn = policy_mapping_fn,
            device = self.device, 
            episode_length=self.episode_length
        )

        # map policy id to agent ids controlled by that policy
        self.policy_agents = {
            policy_id: sorted([agent_id for agent_id in self.agent_ids if policy_mapping_fn(agent_id) == policy_id]) for policy_id in self.policies.keys()
        }

        self.policy_obs_dim = {
            policy_id: self.policies[policy_id].obs_dim for policy_id in self.policy_ids}
        self.policy_act_dim = {
            policy_id: self.policies[policy_id].act_dim for policy_id in self.policy_ids}
        self.policy_central_obs_dim = {
            policy_id: self.policies[policy_id].central_obs_dim for policy_id in self.policy_ids}

        num_train_episodes = (self.num_env_steps / self.episode_length) / (self.train_interval_episode)
        
        self.beta_anneal = DecayThenFlatSchedule(
            self.per_beta_start, 1.0, num_train_episodes, decay="linear")

        if self.use_per:
            self.buffer = PrioritizedRecReplayBuffer(
                alpha = self.per_alpha,    
                policy_info = self.policy_info,
                policy_agents = self.policy_agents,
                buffer_size = self.buffer_size,
                episode_length = self.episode_length,
                use_same_share_obs = self.use_same_share_obs,
                use_avail_acts = self.use_avail_acts,
                use_reward_normalization = self.use_reward_normalization
            )
        else:
            self.buffer = RecReplayBuffer(
                policy_info = self.policy_info,
                policy_agents = self.policy_agents,
                buffer_size = self.buffer_size,
                episode_length = self.episode_length,
                use_same_share_obs = self.use_same_share_obs,
                use_avail_acts = self.use_avail_acts,
                use_reward_normalization = self.use_reward_normalization
            )
    
    def run(self):
        """Collect a training episode and perform appropriate training, saving, logging, and evaluation steps."""
        # collect data
        self.trainer.prep_rollout()
        env_info = self.collect_rollout(explore=True, training_episode=True, warmup=False) 
        for k, v in env_info.items():
            self.env_infos[k].append(v)

        # train
        if ((self.num_episodes_collected - self.last_train_episode) / self.train_interval_episode) >= 1 or self.last_train_episode == 0:
            self.train() 
            self.total_train_steps += 1
            self.last_train_episode = self.num_episodes_collected

        # eval
        if self.use_eval and ((self.total_env_steps - self.last_eval_T) / self.eval_interval) >= 1 or self.total_env_steps >= self.num_env_steps:
            test_rewards = self.eval()
            env_info["Test_Rewards"] = test_rewards
            self.last_eval_T = self.total_env_steps
        
        if self.use_wandb:
            wandb.log(env_info)

        return self.total_env_steps
    
    def warmup(self, num_warmup_episodes):
        """
        Fill replay buffer with enough episodes to begin training.

        :param: num_warmup_episodes (int): number of warmup episodes to collect.
        """
        self.trainer.prep_rollout()
        warmup_rewards = []
        for _ in tqdm(range((num_warmup_episodes // self.num_envs)), desc="warm up..", ncols=70):
            env_info = self.collect_rollout(explore=True, training_episode=False, warmup=True)
            warmup_rewards.append(env_info['average_episode_rewards'])
        warmup_reward = np.mean(warmup_rewards)
        print("warmup average episode rewards: {}".format(warmup_reward))

    def batch_train(self):
        """Do a gradient update for all policies."""
        self.trainer.prep_training()

        # gradient updates
        self.train_infos = []
        update_actor = False
        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            update_method = self.trainer.shared_train_policy_on_batch if self.use_same_share_obs else self.trainer.cent_train_policy_on_batch
            
            train_info, new_priorities, idxes = update_method(p_id, sample)
            update_actor = train_info['update_actor']

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        if self.use_soft_update and update_actor:
            for pid in self.policy_ids:
                self.policies[pid].soft_target_updates()
        else:
            if ((self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode) >= 1:
                for pid in self.policy_ids:
                    self.policies[pid].hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected

    def batch_train_q(self):
        """Do a q-learning update to policy (used for QMix and VDN)."""
        self.trainer.prep_training()
        # gradient updates
        self.train_infos = []

        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
               
                sample = self.buffer.sample(self.batch_size)
            train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(sample) 

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        if self.use_soft_update:
            self.trainer.soft_target_updates()
        else:
            if (self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode >= 1:
                self.trainer.hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected


    def log(self):
        """Log relevent training and rollout colleciton information.."""
        raise NotImplementedError

    def log_clear(self):
        """Clear logging variables so they do not contain stale information."""
        raise NotImplementedError

    def log_env(self, env_info, suffix=None):
        """
        Log information related to the environment.
        :param env_info: (dict) contains logging information related to the environment.
        :param suffix: (str) optional string to add to end of keys in env_info when logging. 
        """
        for k, v in env_info.items():
            if len(v) > 0:
                v = np.mean(v)
                suffix_k = k if suffix is None else suffix + k 
                print(suffix_k + " is " + str(v))


    def log_train(self, policy_id, train_info):
        """
        Log information related to training.
        :param policy_id: (str) policy id corresponding to the information contained in train_info.
        :param train_info: (dict) contains logging information related to training.
        """
        for k, v in train_info.items():
            policy_k = str(policy_id) + '/' + k


    def collect_rollout(self):
        """Collect a rollout and store it in the buffer."""
        raise NotImplementedError # 해당 메서드가 아직 구현되지 않았음을 의미하는 에러 
    
    def obs_sharing(self, obs: List) -> np.array:
        share_obs = np.array(obs).reshape(-1)
        share_obs_list = np.array([share_obs for _ in range(self.num_agents)]) 
        return share_obs_list
    
    def convert_each_rewards(self, rewards):
        converted_rewards = [[reward] for reward in rewards]
        return converted_rewards
    
    def convert_sum_rewards(self, rewards):
        converted_rewards = [[sum(rewards)] for _ in range(self.num_agents)]
        return converted_rewards

    def eval(self):
        """Collect episodes to evaluate the policy."""
        self.trainer.prep_rollout()
        test_rewards = 0.0
        for _ in range(self.args.num_eval_episodes):
            test_rewards += self.collect_rollout(explore=False, training_episode=False, warmup=False)

        return test_rewards