import numpy as np
import torch
import time 
from runner.shared.base_runner import RecRunner

class MAGYM_Runner(RecRunner):
    def __init__(self, config):
        """Runner class for the StarcraftII environment (SMAC). See parent class for more information."""
        super().__init__(config)

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

        last_acts_batch = np.zeros((self.num_envs * len(self.policy_agents[p_id]), self.act_dim), dtype=np.float32)
        rnn_states_batch = np.zeros((self.num_envs * len(self.policy_agents[p_id]), self.hidden_size), dtype=np.float32)

        # init
        episode_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_share_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.central_obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_acts = {p_id : np.zeros((self.episode_length, self.num_envs, self.num_agents, self.act_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {p_id : np.zeros((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones = {p_id : np.ones((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones_env = {p_id : np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}
        
        dones: list = [False for _ in range(self.num_agents)]
        # if not explore:
        #     self.train_env.render() 
            
        for step in range(self.episode_length):
            if warmup:
                acts_batch = policy.get_random_actions(obs = np.array(obs))
                _, rnn_states_batch, _ = policy.get_actions(np.array(obs),
                                                            last_acts_batch,
                                                            rnn_states_batch)
                
            else:
                # get actions with exploration noise (eps-greedy/Gaussian)
                acts_batch, rnn_states_batch, _ = policy.get_actions(np.array(obs),
                                                                    last_acts_batch,
                                                                    rnn_states_batch,
                                                                    t_env=self.total_env_steps,
                                                                    explore=explore)
        
            acts_batch = acts_batch if isinstance(acts_batch, np.ndarray) else acts_batch.cpu().detach().numpy()
            rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
            last_acts_batch = acts_batch

            env_acts = np.argmax(acts_batch, axis=1, keepdims= True)
        
            if not np.all(dones):
                next_obs, rewards, dones, infos = env.step(env_acts)
            else: 
                rewards = [0.0 for _ in range(self.num_agents)]

            next_share_obs = self.obs_sharing(obs = next_obs)
            rewards = self.process_reward_type(rewards)

            # if not explore:
            #     self.train_env.render() 
            #     time.sleep(0.005)

            if training_episode or warmup:
                self.total_env_steps += self.num_envs

            dones_env = np.all(np.array([dones]), axis=1)

            episode_obs[p_id][step] = obs
            episode_share_obs[p_id][step] = share_obs
            episode_acts[p_id][step] = env_acts
            episode_rewards[p_id][step] = rewards

            episode_dones[p_id][step] = np.array([[done] for done in dones])
            episode_dones_env[p_id][step] = dones_env

            obs = next_obs
            share_obs = next_share_obs

            assert self.num_envs == 1, ("only one env is support here.")

        episode_obs[p_id][step] = obs
        episode_share_obs[p_id][step] = share_obs

        if explore:
            self.num_episodes_collected += self.num_envs
            # push all episodes collected in this rollout step to the buffer
            self.buffer.insert(self.num_envs,
                               episode_obs,
                               episode_share_obs,
                               episode_acts,
                               episode_rewards,
                               episode_dones,
                               episode_dones_env)
        if not (explore or training_episode or warmup):
            env_info['Test Rewards'] = np.sum(episode_rewards[p_id][:, 0, 0, 0])
        else:
            env_info['average_episode_rewards'] = np.sum(episode_rewards[p_id][:, 0, 0, 0])
        return env_info
    
    # def log(self):
    #     """See parent class."""
        
    #     print(
    #         "\nEnv {} Algo {} Exp {} runs total num timesteps {}/{}."
    #           .format(self.env_name,
    #                   self.algorithm_name,
    #                   self.args.experiment_name,
    #                   self.total_env_steps,
    #                   self.num_env_steps,
    #                   )
    #         )

    #     for p_id, train_info in zip(self.policy_ids, self.train_infos):
    #         self.log_train(p_id, train_info)

    #     self.log_clear()

    def log_clear(self):
        """See parent class."""
        self.train_env_infos = {}

        self.train_env_infos['average_episode_rewards'] = []

