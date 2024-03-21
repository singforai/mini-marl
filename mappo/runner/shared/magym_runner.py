import time
import wandb
import numpy as np
from functools import reduce
import torch


from runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class MAGYM_Runner(Runner):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes: int = self.max_episodes // self.n_rollout_threads

        for episode in range(episodes):

            self.train_env.render() if self.use_render else None

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode=episode, episodes=episodes)

            step: int = 0
            dones: list = [False for _ in range(self.num_agents)]

            while not all(dones):

                values, actions, action_log_probs, rnn_states, rnn_states_critic = (
                    self.collect(step=step)
                )

                step += 1

                next_obs, rewards, dones, infos = self.train_env.step(
                    actions[0]
                )  # 일단은 batch_size를 1인 상태에서만 돌아가도록 설정해놓았음

                share_obs = self.obs_sharing(obs=next_obs, n_agents=self.num_agents)
                rewards = self.convert_rewards(rewards)

                data = (
                    next_obs,
                    share_obs,
                    rewards,
                    np.array([dones]),
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

                if self.use_render:
                    self.train_env.render()
                    time.sleep(self.sleep_second)

            self.compute()
            train_infos = self.train()

            # total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            self.train_env.reset()

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                eval_results = self.eval(episode=episode)
                train_infos["test_score"] = np.mean(eval_results)

            self.log_train(train_infos=train_infos)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                accumulated_rewards = train_infos["episode_rewards"]
                print(
                    f"Algo {self.algorithm_name} Exp {self.experiment_name} updates {episode}/{episodes} episodes, Accumulated Rewards {accumulated_rewards}"
                )

            if self.use_wandb:
                wandb.log(train_infos)

    def obs_sharing(self, obs, n_agents):
        obs = sum(obs, [])
        share_obs = [obs for _ in range(n_agents)]
        return share_obs

    def convert_rewards(self, rewards):
        converted_rewards = [[reward] for reward in rewards]
        return converted_rewards

    def warmup(self):
        # reset env
        # obs, share_obs, available_actions = self.train_env.reset()
        obs = self.train_env.reset()
        # replay buffer
        if self.use_centralized_V:
            share_obs = self.obs_sharing(obs=obs, n_agents=self.num_agents)
        else:
            share_obs = obs

        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic = (
            self.trainer.policy.get_actions(
                obs=np.concatenate(self.buffer.obs[step]),
                cent_obs=np.concatenate(self.buffer.share_obs[step]),
                rnn_states_actor=np.concatenate(self.buffer.rnn_states[step]),
                rnn_states_critic=np.concatenate(self.buffer.rnn_states_critic[step]),
                masks=np.concatenate(self.buffer.masks[step]),
            )
        )

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_state_critic), self.n_rollout_threads)
        )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        dones_env = np.all(dones, axis=1)
        rnn_states[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        rnn_states_critic[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                *self.buffer.rnn_states_critic.shape[3:],
            ),
            dtype=np.float32,
        )

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        active_masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32
        )
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        ),  # active_mask

    def log_train(self, train_infos):
        train_infos["episode_rewards"] = np.sum(self.buffer.rewards)

    @torch.no_grad()
    def eval(self, episode):
        eval_total_rewards = []

        for test_number in range(self.eval_episodes):

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
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )

            self.trainer.prep_rollout()

            eval_episode_rewards: float = 0.0
            eval_dones: list = [False for _ in range(self.num_agents)]
            eval_step: int = 0

            while not all(eval_dones):
                eval_actions, eval_rnn_states = self.trainer.policy.act(
                    obs=np.concatenate([eval_obs]),
                    rnn_states_actor=np.concatenate(eval_rnn_states),
                    masks=np.concatenate(eval_masks),
                    deterministic=True,
                )
                eval_actions = np.array(
                    np.split(_t2n(eval_actions), self.n_eval_rollout_threads)
                )
                eval_rnn_states = np.array(
                    np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
                )

                eval_next_obs, eval_rewards, eval_dones, eval_infos = (
                    self.eval_env.step(eval_actions[0])
                )

                eval_episode_rewards += sum(eval_rewards)
                eval_obs = eval_next_obs

            eval_total_rewards.append(eval_episode_rewards)

            self.eval_env.reset()

        return eval_total_rewards
