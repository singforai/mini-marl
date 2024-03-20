import torch
import random

import numpy as np

from replay_buffer.sumtree import SumTree


class Prioritized_Experience_Replay():
    def __init__(self, args):

        self.use_step_weight: bool = args.use_step_weight

        self.capacity: int = args.buffer_limit

        self.step_weight: float = args.step_weight
        self.eps: float = args.eps
        self.alpha: float = args.alpha
        self.beta: float = args.beta

        self.sum_tree = SumTree(self.capacity)
        self.priority_buffer = self.sum_tree.priority_tree
        self.buffer = self.sum_tree.buffer

        if args.update_alpha_beta:
            self.alpha_increment_per_sampling = (
                1-self.alpha)/(args.max_episodes * args.update_iter)
            self.beta_increment_per_sampling = (
                1-self.beta)/(args.max_episodes * args.update_iter)
        else:
            self.alpha_increment_per_sampling = 0
            self.beta_increment_per_sampling = 0

    def _get_priority(self, td_error):
        return (td_error + self.eps) ** self.alpha

    def collect_sample(self, sample, td_error, warm_up):
        priority = self._get_priority(td_error)
        n_buffer = self.sum_tree.add(priority, sample)
        return n_buffer if warm_up else None

    def sample(self, batch_size, _chunk_size):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        priority_idxs, priority_values = [], []

        segment = self.sum_tree.total() / batch_size

        self.alpha: float = np.min(
            [1., self.alpha + self.alpha_increment_per_sampling])
        self.beta: float = np.min(
            [1., self.beta + self.beta_increment_per_sampling])

        for interval in range(batch_size):
            segment_left: float = segment * interval
            segment_right: float = segment * (interval + 1)
            segment_sample: float = random.uniform(segment_left, segment_right)

            priority_idx, priority_value, dataIdx = self.sum_tree.get(
                segment_sample)

            priority_values.append(priority_value)
            priority_idxs.append(priority_idx)

            data_chunk = self.buffer[dataIdx]
            state_chunk, action_chunk, reward_chunk, s_prime_chunk, done_chunk = data_chunk
            s_lst.append(state_chunk)
            a_lst.append(action_chunk)
            r_lst.append(reward_chunk)
            s_prime_lst.append(s_prime_chunk)
            done_lst.append(done_chunk)

        if self.use_step_weight:
            self.sum_tree.priority_tree = self.step_weight*self.sum_tree.priority_tree

        sampling_probabilities = priority_values / self.sum_tree.total()
        importance_sampling_weight = np.power(
            self.capacity * sampling_probabilities, -self.beta)
        importance_sampling_weight /= importance_sampling_weight.max()

        n_agents, obs_size = len(s_lst[0][0]), len(s_lst[0][0][0])
        return torch.tensor(s_lst, dtype=torch.float).view(batch_size, _chunk_size, n_agents, obs_size), \
            torch.tensor(a_lst, dtype=torch.float).view(batch_size, _chunk_size, n_agents), \
            torch.tensor(r_lst, dtype=torch.float).view(batch_size, _chunk_size, n_agents), \
            torch.tensor(s_prime_lst, dtype=torch.float).view(batch_size, _chunk_size, n_agents, obs_size), \
            torch.tensor(done_lst, dtype=torch.float).view(
                batch_size, _chunk_size, 1), priority_idxs, torch.tensor(importance_sampling_weight, dtype=torch.float).view(batch_size, 1)

    def update(self, priority_idx, new_td_error):
        priority = self._get_priority(new_td_error)
        self.sum_tree.update(priority_idx, priority)
