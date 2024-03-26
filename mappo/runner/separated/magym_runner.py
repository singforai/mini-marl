import time
import wandb
import torch

import numpy as np

from runner.separated.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()


class MAGYM_Runner(Runner):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = self.max_episodes // self.n_rollout_threads

        for episode in range(episodes):

            self.train_env.render() if self.use_render else None

            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)
            
            step: int = 0
            dones: list = [False for _ in range(self.num_agents)]

            while not all(dones):
                (
                    action_values,
                    actions,
                    action_log_probs,
                    rnn_states_actor,
                    rnn_states_critic,
                    actions_env
                ) = self.collect(step=step)

                # Obser reward and next obs
                next_obs, rewards, dones, infos = self.train_env.step(
                    actions[0]
                    )
                
                rewards = self.convert_rewards(rewards)
                next_share_obs = self.process_obs_type(obs = next_obs)

                data = (
                    next_obs,
                    next_share_obs,
                    rewards,
                    np.array([dones]),
                    infos,
                    action_values,
                    actions,
                    action_log_probs,
                    rnn_states_actor,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

                if self.use_render:
                    self.train_env.render()
                    time.sleep(self.sleep_second)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "Env {} Algo {} Exp {} updates {}/{} episodes, FPS {}.\n".format(
                        self.env_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        int( episode/ (end - start)),
                    )
                )

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                eval_result: float = self.eval()

            self.log_train(train_infos = train_infos, eval_result = eval_result)

            self.train_env.reset()
    
    def convert_rewards(self, rewards):
        converted_rewards = [[reward] for reward in rewards]
        return converted_rewards

    def warmup(self):
        # reset env
        
        obs = self.train_env.reset()
        share_obs_list = self.process_obs_type(obs = obs)
      

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs_list[agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        action_values = []
        actions = []

        rnn_states = []
        rnn_states_critic = []

        temp_actions_env = []
        action_log_probs = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[agent_id].policy.get_actions(
                obs = self.buffer[agent_id].obs[step],
                cent_obs = self.buffer[agent_id].share_obs[step],
                rnn_states_actor = self.buffer[agent_id].rnn_states[step],
                rnn_states_critic = self.buffer[agent_id].rnn_states_critic[step],
                masks = self.buffer[agent_id].masks[step],
            )
            
            # [agents, envs, dim]
            action_values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.train_env.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                for i in range(self.train_env.action_space[agent_id].shape):
                    uc_action_env = np.eye(
                        self.train_env.action_space[agent_id].high[i] + 1
                    )[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.train_env.action_space[agent_id].__class__.__name__ == "Discrete":
                action_env = np.squeeze(
                    np.eye(self.train_env.action_space[agent_id].n)[action], 1
                )
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

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
            actions_env,
        )

    def insert(self, data):
        (
            next_obs,
            next_share_obs,
            rewards,
            dones,
            infos,
            action_values,
            actions,
            action_log_probs,
            rnn_states_actor,
            rnn_states_critic,
        ) = data

        rnn_states_actor[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(
                next_share_obs[agent_id],
                next_obs[agent_id],
                rnn_states_actor[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                action_values[:, agent_id],
                rewards[agent_id],
                masks[:, agent_id],
            )

    @torch.no_grad()
    def eval(self):
        eval_obs = self.eval_env.reset()
        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), 
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
