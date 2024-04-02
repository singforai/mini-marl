import numpy as np
import torch
import time 
from runner.separated.base_runner import RecRunner

class MAGYM_Runner(RecRunner):
    def __init__(self, config):
        """Runner class for the StarcraftII environment (SMAC). See parent class for more information."""
        super(MAGYM_Runner, self).__init__(config)
        # fill replay buffer with random actions
        self.warmup(self.args.num_warmup_episodes)
        self.log_clear()

    
    @torch.no_grad()
    def collect_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        Collect a rollout and store it in the buffer. All agents share a single policy.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.train_env if training_episode or warmup else self.eval_env

        obs = env.reset()
        share_obs = self.obs_sharing(obs=obs)

        self.act_dim = policy.output_dim
        
        last_acts_batch = {p_id : np.zeros((self.num_envs * len(self.policy_agents[p_id]), self.act_dim), dtype=np.float32) for p_id in self.policy_ids}
        rnn_states_batch = {p_id : np.zeros((self.num_envs * len(self.policy_agents[p_id]) , self.hidden_size), dtype=np.float32) for p_id in self.policy_ids}

        # init
        episode_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs,1, policy.obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_share_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, 1, policy.central_obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_acts = {p_id : np.zeros((self.episode_length, self.num_envs, 1, self.act_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {p_id : np.zeros((self.episode_length, self.num_envs, 1, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones = {p_id : np.ones((self.episode_length, self.num_envs, 1, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones_env = {p_id : np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}

        dones: list = [False for _ in range(self.num_agents)]
        for step in range(self.episode_length):
            # get actions for all agents to step the env


            acts_batchs, rnn_states_batchs = [], [] # 2x5, 2x64
            for idx, agent_id in enumerate(self.policy_ids):
                obs =np.array([obs[idx-1]])
                if warmup:
                    # completely random actions in pre-training warmup phase
                    acts_batch = self.policies[agent_id].get_random_actions(obs = obs)
                    _, rnn_states_batch[agent_id], _ = self.policies[agent_id].get_actions(
                        obs = obs,
                        prev_actions = last_acts_batch[agent_id],
                        rnn_states = rnn_states_batch[agent_id],
                    )
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    acts_batch, rnn_states_batch[agent_id], _ = policy.get_actions(
                        obs = obs,
                        prev_actions = last_acts_batch[agent_id],
                        rnn_states = rnn_states_batch[agent_id],
                        t_env=self.total_env_steps,
                        explore=explore
                )
                acts_batch_list = acts_batch if isinstance(acts_batch, np.ndarray) else acts_batch.cpu().detach().tolist()
                rnn_states_batch_list = rnn_states_batch[agent_id] if isinstance(rnn_states_batch[agent_id], np.ndarray) else rnn_states_batch[agent_id].cpu().detach().tolist()
                last_acts_batch[agent_id] = acts_batch_list
                acts_batchs.append(acts_batch_list.tolist()[0])
                rnn_states_batchs.append(rnn_states_batch_list)

            env_acts = np.argmax(acts_batchs, axis=1, keepdims= True)

            print(env_acts, "===")

            # env step and store the relevant episode information
            if not np.all(dones):
                next_obs, rewards, dones, infos = env.step(env_acts)
            else: 
                rewards = [0.0 for _ in range(self.num_agents)]

            next_share_obs = self.obs_sharing(obs = next_obs)
            rewards = self.process_reward_type(rewards)
            
            if training_episode or warmup:
                self.total_env_steps += self.num_envs

            dones_env = np.all(np.array([dones]), axis=1)

            for idx, p_id in enumerate(self.policy_info.keys()):
                episode_obs[p_id][step] = obs[idx-1]
                episode_share_obs[p_id][step] = share_obs[idx-1]
                episode_acts[p_id][step] = env_acts[idx-1]
                episode_rewards[p_id][step] = rewards[idx-1]
                # here dones store agent done flag of the next step
                episode_dones[p_id][step] = np.array([[done] for done in dones])[idx-1]
                episode_dones_env[p_id][step] = dones_env[idx-1]

            obs = next_obs
            share_obs = next_share_obs

            assert self.num_envs == 1, ("only one env is support here.")

        if explore:
            self.num_episodes_collected += self.num_envs
            # push all episodes collected in this rollout step to the buffer
            for p_id in self.policy_info.keys():
                self.buffer.policy_buffers[p_id].insert(
                    num_insert_episodes = self.num_envs,
                    obs = episode_obs[p_id],
                    share_obs = episode_share_obs[p_id],
                    acts = episode_acts[p_id],
                    rewards = episode_rewards[p_id],
                    dones = episode_dones[p_id],
                    dones_env = episode_dones_env[p_id],
                )

        if not (explore or training_episode or warmup):
            test_rewards = 0
            for p_id in self.policy_info.keys():
                test_rewards += np.sum(episode_rewards[p_id])
            return test_rewards
        
        else:   
            env_info['average_episode_rewards'] = 0
            for p_id in self.policy_info.keys():
                env_info['average_episode_rewards'] += np.sum(episode_rewards[p_id])
            return env_info

    def log(self):
        """See parent class."""

        for p_id, train_info in zip(self.policy_ids, self.train_infos):
            self.log_train(p_id, train_info)

        self.log_env(self.env_infos)
        self.log_clear()

    def log_clear(self):
        """See parent class."""
        self.env_infos = {}

        self.env_infos['average_episode_rewards'] = []
        self.env_infos['win_rate'] = []
