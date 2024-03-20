import copy
import torch
from abc import abstractmethod

import torch.nn.functional as F


class Train_Target:
    def __init__(self, Replay_buffer, behavior_network, target_network, args, device):
        self.device = device
        self.replay_buffer = Replay_buffer
        self.behavior_network = behavior_network
        self.target_network = target_network

        self.batch_size: int = args.batch_size
        self.update_iter: int = args.update_iter
        self.chunk_size: int = args.chunk_size if args.use_recurrent else 1
        self.gamma: float = args.gamma
        self.grad_clip_norm: float = args.grad_clip_norm
        
        self.device = device

    def sample_chunk(self, batch_size, chunk_size):
        states, actions, rewards, next_states, dones, priority_idx, is_weight = self.replay_buffer.sample(
            batch_size, chunk_size)
        return states, actions.to(self.device), rewards.to(self.device), next_states, dones.to(self.device), priority_idx, is_weight.to(self.device)

    def initialize_hidden(self, batch_size1, batch_size2, batch_size3=None):
        hidden = self.behavior_network.init_hidden(
            batch_size=batch_size1).to(self.device)
        target_hidden = self.target_network.init_hidden(
            batch_size=batch_size2).to(self.device)
        if batch_size3 == None:
            return hidden, target_hidden
        else:
            double_hidden = self.target_network.init_hidden(
                batch_size=batch_size3).to(self.device)
            return hidden, target_hidden, double_hidden

    def extract_step(self, step, states, actions, rewards, next_states, dones):
        return states[:, step, :, :], actions[:, step, :].unsqueeze(dim=-1).long(), rewards[:, step, :], next_states[:, step, :, :], dones[:, step, :]

    def sum_behavior_value(self, state, hidden, action):
        behavior_q, next_hidden = self.behavior_network(
            obs=state.to(self.device), hidden=hidden)
        max_behavior_q = behavior_q.gather(dim=2, index=action).squeeze(dim=-1)
        return max_behavior_q.sum(dim=1, keepdims=True), next_hidden


class Target_Dqn(Train_Target):
    def __init__(self, Replay_buffer, behavior_network, target_network, args, device):
        super().__init__(Replay_buffer, behavior_network, target_network, args, device)
        
        self.device = device

    def train(self, target_network, optimizer, epsilon):
        iter_loss = 0
        for _ in range(self.update_iter):
            states, actions, rewards, next_states, dones, priority_idx, is_weight = super().sample_chunk(batch_size=self.batch_size,
                                                                                                         chunk_size=self.chunk_size)
            hidden, target_hidden = super().initialize_hidden(batch_size1=self.batch_size,
                                                              batch_size2=self.batch_size)
            loss = 0.0
            for step in range(self.chunk_size):
                state, action, reward, next_state, done = super().extract_step(
                    step, states, actions, rewards, next_states, dones)
                sum_max_behavior_q, next_hidden = super().sum_behavior_value(state=state.to(self.device),
                                                                             hidden=hidden,
                                                                             action=action)

                target_q, next_target_hidden = target_network(
                    obs=next_state.to(self.device), hidden=target_hidden.detach())
                max_target_q = target_q.max(dim=2)[0]
                sum_max_target_q = max_target_q.sum(dim=1, keepdims=True)
                
                target_values = is_weight*(reward + self.gamma * (1 - done) *
                                           sum_max_target_q).sum(dim=1, keepdims=True)

                loss += F.mse_loss(target_values.detach(), sum_max_behavior_q)

                done_mask = done.squeeze(-1).bool()
                next_hidden[done_mask], next_target_hidden[done_mask] = super().initialize_hidden(batch_size1=len(next_hidden[done_mask]),
                                                                                                  batch_size2=len(next_target_hidden[done_mask]))

                hidden = next_hidden
                target_hidden = next_target_hidden

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.behavior_network.parameters(
            ), max_norm=self.grad_clip_norm, norm_type=2)
            optimizer.step()

            iter_loss += loss.detach()

            new_td_error = abs(target_values-sum_max_behavior_q)
            for index in range(self.batch_size):
                idx = priority_idx[index]
                self.replay_buffer.update(idx, new_td_error[index])

        return iter_loss/self.update_iter


class Target_Double_Dqn(Train_Target):
    def __init__(self, Replay_buffer, behavior_network, target_network, args, device):
        super().__init__(Replay_buffer, behavior_network, target_network, args, device)

    def train(self, target_network, optimizer, epsilon):
        iter_loss: float = 0
        for _ in range(self.update_iter):
            double_network = copy.deepcopy(self.behavior_network)
            states, actions, rewards, next_states, dones, priority_idx, is_weight = super().sample_chunk(batch_size=self.batch_size,
                                                                                                         chunk_size=self.chunk_size)
            hidden, target_hidden = super().initialize_hidden(batch_size1=self.batch_size,
                                                              batch_size2=self.batch_size)
            double_hidden = double_network.init_hidden(
                batch_size=self.batch_size).to(self.device)

            loss: float = 0.0
            for step in range(self.chunk_size):
                state, action, reward, next_state, done = super().extract_step(
                    step, states, actions, rewards, next_states, dones)
                sum_max_behavior_q, next_hidden = super().sum_behavior_value(state=state.to(self.device),
                                                                             hidden=hidden,
                                                                             action=action)
                target_q, next_target_hidden = target_network(
                    next_state.to(self.device), target_hidden.detach())
                action, next_double_hidden, _ = double_network.sample_action(
                    next_state.to(self.device), double_hidden, epsilon)
                double_q = target_q.gather(
                    dim=2, index=action.long().unsqueeze(dim=-1)).squeeze(dim=-1)
                sum_double_q = double_q.sum(dim=1, keepdims=True)

                target_values = is_weight * \
                    (reward + self.gamma * (1 - done) *
                     sum_double_q).sum(dim=1, keepdims=True)
                loss += F.mse_loss(target_values.detach(), sum_max_behavior_q)
                done_mask = done.squeeze(-1).bool()
                next_hidden[done_mask], next_target_hidden[done_mask], next_double_hidden[done_mask] = super().initialize_hidden(
                    batch_size1=len(next_hidden[done_mask]),
                    batch_size2=len(next_target_hidden[done_mask]),
                    batch_size3=len(next_double_hidden[done_mask]))
                hidden = next_hidden
                target_hidden = next_target_hidden
                double_hidden = next_double_hidden

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.behavior_network.parameters(
            ), max_norm=self.grad_clip_norm, norm_type=2)
            optimizer.step()

            iter_loss += loss
            new_td_error = abs(target_values-sum_max_behavior_q)
            for index in range(self.batch_size):
                idx = priority_idx[index]
                self.replay_buffer.update(idx, new_td_error[index])
        return iter_loss / self.update_iter
