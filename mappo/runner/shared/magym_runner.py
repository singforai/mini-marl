import wandb
import torch
import multiprocessing

import numpy as np

from tqdm import tqdm
from typing import List, Dict


from runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()

class MAGYM_Runner(Runner):
    def __init__(self, config):
        super().__init__(config)

    # def process_batch(self, queue, batch_action):
    #     next_obs, rewards, dones, _ = self.train_env.step(batch_action)
    #     step_result: Dict = {
    #         "next_obs": next_obs, 
    #         "rewards": rewards, 
    #         "dones": dones
    #         }
    #     queue.put(step_result)

    def run(self):
        self.warmup()

        for episode in tqdm(range(self.max_episodes ), desc="training" , ncols=70):

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode=episode, episodes=self.max_episodes)

            step: int = 0
            dones: list = [[False for _ in range(self.num_agents)] for _ in range(self.batch_size)]

            for step in range(self.episode_length):

                (
                action_values, 
                actions, 
                action_log_probs, 
                rnn_states_actor, 
                rnn_states_critic
                ) = self.collect(step=step)
                

                next_obs_batch, rewards_batch, dones_batch = [], [], []
                for batch_action in actions:
                    next_obs, rewards, dones, _ = self.train_env.step(batch_action)
                    next_obs_batch.append(next_obs)
                    rewards_batch.append(rewards)
                    dones_batch.append(dones)
                
                # procs: List[object] = []

                # for batch_action in actions:
                #     process = multiprocessing.Process(target = self.process_batch, args = (self.queue, batch_action))
                #     process.start()
                #     procs.append(process)

                # for each_procs in procs:
                #     each_procs.join()

                # next_obs_batch, rewards_batch, dones_batch = [], [], []
                # while not self.queue.empty():
                #     batch_result = self.queue.get()
                #     next_obs_batch.append(batch_result["next_obs"])
                #     rewards_batch.append(batch_result["rewards"])
                #     dones_batch.append(batch_result["dones"])

                
                next_share_obs_batch = self.process_obs_type(obs=next_obs_batch)
                rewards_batch = self.process_reward_type(rewards_batch = rewards_batch)


                data = (
                    next_obs_batch,
                    next_share_obs_batch,
                    rewards_batch,
                    np.array([dones_batch]),
                    action_values,
                    actions,
                    action_log_probs,
                    rnn_states_actor,
                    rnn_states_critic,
                )

                self.insert(data = data)
                step += 1

            self.compute()
            train_infos = self.train()
            self.train_env.reset()

            if episode % self.eval_interval == 0 and self.use_eval:
                eval_results = self.eval()
                train_infos["Test_Rewards"] = eval_results

            if self.use_wandb:
                wandb.log(train_infos)

    def warmup(self):
        obs = self.train_env.reset()
        share_obs = self.process_obs_type(obs=obs, warm_up = True)
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

        action_values = np.array(np.split(_t2n(value), self.batch_size))
        actions = np.array(np.split(_t2n(action), self.batch_size))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.batch_size)
        )
        rnn_states = np.array(np.split(_t2n(rnn_state), self.batch_size))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_state_critic), self.batch_size)
        )

        return action_values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        (
            next_obs,
            next_share_obs,
            rewards,
            dones,
            action_values,
            actions,
            action_log_probs,
            rnn_states_actor,
            rnn_states_critic,
        ) = data

        dones_env = np.all(dones, axis=2).reshape(-1)
        dones = dones.reshape(self.batch_size, self.num_agents)
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

        masks = np.ones((self.batch_size, self.num_agents, 1), dtype=np.float32)        
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        active_masks = np.ones(
            (self.batch_size, self.num_agents, 1), dtype=np.float32
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
