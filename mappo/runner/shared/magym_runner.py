import time
import wandb
import numpy as np
import torch


from runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class MAGYM_Runner(Runner):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        self.warmup()

        episodes: int = self.max_episodes // self.n_rollout_threads

        for episode in range(episodes):

            self.train_env.render() if self.use_render else None

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode=episode, episodes=episodes)

            step: int = 0
            dones: list = [False for _ in range(self.num_agents)]

            while not all(dones):
                (
                action_values, 
                actions, 
                action_log_probs, 
                rnn_states_actor, 
                rnn_states_critic
                ) = self.collect(step=step)

                next_obs, rewards, dones, infos = self.train_env.step(
                    actions[0]
                )

                next_share_obs = self.process_obs_type(obs=next_obs)
                rewards = self.convert_rewards(rewards)

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
                self.insert(data = data)

                if self.use_render:
                    self.train_env.render()
                    time.sleep(self.sleep_second)

                

                step += 1

            #self.process_mask(step = step)

            self.compute()
            train_infos = self.train()
            self.train_env.reset()
            # eval
            # if episode % self.eval_interval == 0 and self.use_eval:
            #     eval_results = self.eval()
            #     train_infos["Test_Rewards"] = eval_results

            # log information
            if episode % self.log_interval == 0:
                print(
                    f"Algo {self.algorithm_name} Exp {self.experiment_name} updates {episode}/{episodes} episodes"
                )
            if self.use_wandb:
                wandb.log(train_infos)

    def process_mask(self, step: int):
        self.buffer.masks[step:] = 0.0

    def convert_rewards(self, rewards):
        converted_rewards = [[reward] for reward in rewards]
        return converted_rewards

    def warmup(self):
        obs = self.train_env.reset()
        share_obs = self.process_obs_type(obs=obs)

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

        action_values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_state_critic), self.n_rollout_threads)
        )

        return action_values, actions, action_log_probs, rnn_states, rnn_states_critic

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

        dones_env = np.all(dones, axis=1)

        rnn_states_actor[dones_env == True] = np.zeros(
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
        self.buffer.insert(
            share_obs = next_share_obs,
            obs = next_obs,
            rnn_states_actor = rnn_states_actor,
            rnn_states_critic = rnn_states_critic,
            actions = actions,
            action_log_probs = action_log_probs,
            value_preds = action_values,
            rewards = rewards,
            masks = masks
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
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        self.trainer.prep_rollout()

        eval_episode_rewards: float = 0.0
        eval_dones: list = [False for _ in range(self.num_agents)]

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

            eval_next_obs, eval_rewards, eval_dones, _ = self.eval_env.step(
                eval_actions[0]
            )

            eval_episode_rewards += sum(eval_rewards)
            eval_obs = eval_next_obs

        self.eval_env.reset()

        return eval_episode_rewards
