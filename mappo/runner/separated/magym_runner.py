import time
import torch

import numpy as np

from tqdm import tqdm

from runner.separated.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()


class MAGYM_Runner(Runner):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        self.warmup()  

        max_episodes = self.max_episodes // self.sampling_batch_size

        for episode in tqdm(range(max_episodes ), desc="training" , ncols=70):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, max_episodes)
            
            init_dones: list = [[False for _ in range(self.num_agents)] for _ in range(self.sampling_batch_size)]

            for step in range(self.episode_length):
                (
                    action_values,
                    actions,
                    action_log_probs,
                    rnn_states_actor,
                    rnn_states_critic,
                ) = self.collect(step=step)

                next_obs_batch, rewards_batch, dones_batch = [], [], []
                for idx, batch_action in enumerate(actions):
                    if all(init_dones[idx]):
                        rewards = [0.0 for _ in range(self.num_agents)]
                    else: 
                        next_obs, rewards, dones, _ = self.train_env[idx].step(batch_action)
                    next_obs_batch.append(next_obs)
                    rewards_batch.append(rewards)
                    dones_batch.append(dones)
                
                next_share_obs_batch = self.process_obs_type(obs=next_obs_batch)
                rewards_batch = self.process_reward_type(rewards_batch = rewards_batch)
                
                data = (
                    np.array(next_obs_batch),
                    next_share_obs_batch,
                    np.array(rewards_batch),
                    np.array([dones_batch]),
                    action_values,
                    actions,
                    action_log_probs,
                    rnn_states_actor,
                    rnn_states_critic,
                )  
                self.insert(data)
                
                init_dones = dones_batch

            self.compute()
            train_infos = self.train()
        
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                eval_result: float = self.eval()

            self.log_train(train_infos = train_infos, eval_result = eval_result)

            for idx in range(self.sampling_batch_size):
                self.train_env[idx].reset()
    
    def convert_rewards(self, rewards):
        converted_rewards = [[reward] for reward in rewards]
        return converted_rewards

    def warmup(self):
        # reset env
        
        obs = self.train_env[0].reset()
        share_obs = self.process_obs_type(obs=obs, warm_up = True)
        for agent_id in range(self.num_agents):
                self.buffer[agent_id].obs[0] = obs[agent_id].copy()
                self.buffer[agent_id].share_obs[0] = share_obs[agent_id].copy()
        
    @torch.no_grad()
    def collect(self, step):

        action_values = []
        actions = []
        rnn_states = []
        rnn_states_critic = []
        action_log_probs = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()

            action_value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[agent_id].policy.get_actions(
                obs = self.buffer[agent_id].obs[step],
                cent_obs = self.buffer[agent_id].share_obs[step],
                rnn_states_actor = self.buffer[agent_id].rnn_states[step],
                rnn_states_critic = self.buffer[agent_id].rnn_states_critic[step],
                masks = self.buffer[agent_id].masks[step],
            )

            action_values.append(_t2n(action_value))
            action = _t2n(action)

            actions.append(action)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        action_values = np.array(action_values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return (
            action_values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        )

    def insert(self, data):
        (
            next_obs_batch,
            next_share_obs_batch,
            rewards_batch,
            dones_batch,
            action_values,
            actions,
            action_log_probs,
            rnn_states_actor,
            rnn_states_critic,
        ) = data

        dones_batch = dones_batch.reshape(self.sampling_batch_size, self.num_agents)

        dones_env = np.all(dones_batch, axis=1)

        rnn_states_actor[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(), 
                self.num_agents,
                self.recurrent_N, 
                self.hidden_size
             ),
            dtype=np.float32,
        )
        rnn_states_critic[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(), 
                self.num_agents,
                self.recurrent_N, 
                self.hidden_size,
             ),
            dtype=np.float32,
        )
        masks = np.ones(
            (
                self.sampling_batch_size, 
                self.num_agents, 
                1
             ), 
            dtype=np.float32
            )
        masks[dones_batch == True] = np.zeros(
            (
                (dones_batch == True).sum(), 1),
                dtype=np.float32
            )

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(
                next_share_obs_batch[: ,agent_id],
                next_obs_batch[: ,agent_id],
                rnn_states_actor[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                action_values[:, agent_id],
                rewards_batch[:, agent_id],
                masks[:, agent_id],
            )

    @torch.no_grad()
    def eval(self):
        eval_obs = self.eval_env.reset()
        eval_batch = 1
        eval_rnn_states = np.zeros(
            (
                eval_batch,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (eval_batch, self.num_agents, 1), 
            dtype=np.float32
        )
        eval_dones: list = [False for _ in range(self.num_agents)]
        eval_episode_rewards: float = 0.0

        while not all(eval_dones):
            eval_agent_actions: list = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    obs=torch.tensor([eval_obs[agent_id]]),
                    rnn_states_actor=eval_rnn_states[:, agent_id],
                    masks=eval_masks[:, agent_id],
                    deterministic=True,
                )
                eval_action = eval_action[0].detach().cpu().tolist()
                eval_agent_actions.append(eval_action)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            eval_next_obs, eval_rewards, eval_dones, _ = self.eval_env.step(np.array(eval_agent_actions))

            eval_episode_rewards += sum(eval_rewards)
            eval_obs = eval_next_obs

        self.eval_env.reset()
        return eval_episode_rewards
