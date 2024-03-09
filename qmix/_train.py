import copy
import torch

import torch.nn.functional as F

class Train_dqn:
    def __init__(self, args, device):

        self.batch_size: int = args.batch_size
        self.update_iter: int = args.update_iter
        self.chunk_size: int = args.chunk_size if args.use_recurrent else 1

        self.gamma: float = args.gamma
        self.grad_clip_norm: float = args.grad_clip_norm

        self.device = device

    def train(self, Replay_buffer, behavior_q_net, behavior_mix_net, target_q_net, target_mix_net, optimizer, epsilon):
        for _ in range(self.update_iter):
            states, actions, rewards, next_states, dones, priority_idx, importance_sampling_weight = Replay_buffer.sample(self.batch_size, self.chunk_size)
            target_hidden = target_q_net.init_hidden(batch_size = self.batch_size).to(self.device)
            target_mix_hidden = target_mix_net.init_hidden(batch_size = self.batch_size).to(self.device)
            
            hidden = behavior_q_net.init_hidden(batch_size = self.batch_size).to(self.device)
            mix_hiddens = [torch.empty_like(target_mix_hidden) for _ in range(self.chunk_size + 1)]
            mix_hiddens[0] = behavior_mix_net.init_hidden(self.batch_size)
            loss = 0.0

            for step in range(self.chunk_size):
                state = states[:, step, :, :]
                action = actions[:, step, :].unsqueeze(dim=-1).long()
                reward = rewards[:, step, :]
                next_state = next_states[:, step, :, :]
                done = dones[:, step, :]
                mix_hidden = mix_hiddens[step]

                behavior_q, next_hidden = behavior_q_net(state.to(self.device), hidden)
                max_behavior_q = behavior_q.gather(dim=2, index=action).squeeze(dim=-1)
                sum_mix_q, next_mix_hidden = behavior_mix_net(max_behavior_q.to(self.device), state.to(self.device), mix_hidden.to(self.device))
                #sum_max_behavior_q = max_behavior_q.sum(dim=1, keepdims=True)

                target_q, next_target_hidden = target_q_net(next_state.to(self.device), target_hidden.detach())
                max_target_q = target_q.max(dim=2)[0].squeeze(dim = -1)
                sum_target_mix_q, next_target_mix_hidden = target_mix_net(max_target_q, next_state, target_mix_hidden.detach())
                #sum_max_target_q = max_target_q.sum(dim=1, keepdims=True)
                target_values = importance_sampling_weight * (reward + self.gamma * (1 - done) * sum_target_mix_q).sum(dim=1, keepdims=True)
                
                loss += F.mse_loss(target_values.detach(), sum_mix_q)

                done_mask = done.squeeze(-1).bool()
                next_hidden[done_mask] = behavior_q_net.init_hidden(batch_size=len(next_hidden[done_mask])).to(self.device)
                next_target_hidden[done_mask] = behavior_q_net.init_hidden(batch_size=len(next_target_hidden[done_mask])).to(self.device)
                mix_hiddens[step + 1][~done_mask] = next_mix_hidden[~done_mask].to(self.device)
                mix_hiddens[step + 1][done_mask] = behavior_mix_net.init_hidden(len(next_mix_hidden[step][done_mask])).to(self.device)
                next_target_mix_hidden[done_mask] = target_mix_net.init_hidden(len(next_target_mix_hidden[done_mask])).to(self.device)
                
                hidden = next_hidden
                mix_hidden = next_mix_hidden
                target_hidden = next_target_hidden
                target_mix_hidden = next_target_mix_hidden


            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=behavior_q_net.parameters(), max_norm=self.grad_clip_norm, norm_type=2)
            optimizer.step()

            new_td_error = abs(target_values-sum_mix_q)
            for index in range(self.batch_size):
                idx = priority_idx[index]
                Replay_buffer.update(idx, new_td_error[index])

class Train_double_dqn:
    def __init__(self, Replay_buffer, behavior_q_net, behavior_mix_net, target_q_net, target_mix_net, optimizer, epsilon):

        self.batch_size: int = args.batch_size
        self.update_iter: int = args.update_iter
        self.chunk_size: int = args.chunk_size if args.use_recurrent else 1

        self.gamma: float = args.gamma
        self.grad_clip_norm: float = args.grad_clip_norm

        self.device = device

    def train(self, Replay_buffer, behavior_q_net, target_q_net, optimizer, epsilon):
        for _ in range(self.update_iter):
            double_network = copy.deepcopy(behavior_q_net)

            states, actions, rewards, next_states, dones, priority_idx, importance_sampling_weight = Replay_buffer.sample(self.batch_size, self.chunk_size)
            
            hidden = behavior_q_net.init_hidden(batch_size = self.batch_size).to(self.device)
            double_hidden = double_network.init_hidden(batch_size = self.batch_size).to(self.device)
            target_hidden = target_q_net.init_hidden(batch_size = self.batch_size).to(self.device)

            loss = 0.0
            for step in range(self.chunk_size):
                state = states[:, step, :, :]
                action = actions[:, step, :].unsqueeze(dim=-1).long()
                reward = rewards[:, step, :]
                next_state = next_states[:, step, :, :]
                done = dones[:, step, :]

                behavior_q, next_hidden = behavior_q_net(state.to(self.device), hidden)
                max_behavior_q = behavior_q.gather(dim=2, index=action).squeeze(dim=-1)
                sum_max_behavior_q = max_behavior_q.sum(dim=1, keepdims=True)

                target_q, next_target_hidden = target_q_net(next_state.to(self.device), target_hidden.detach())
                action, next_double_hidden, _ = double_network.sample_action(next_state.to(self.device), double_hidden, epsilon)
                double_q = target_q.gather(dim=2, index = action.long().unsqueeze(dim=-1)).squeeze(dim=-1)
                sum_double_q = double_q.sum(dim=1, keepdims=True)

                target_values = importance_sampling_weight * (reward + self.gamma * (1 - done) * sum_double_q).sum(dim=1, keepdims=True)
                
                loss += F.mse_loss(target_values.detach(), sum_max_behavior_q)

                done_mask = done.squeeze(-1).bool()
                next_hidden[done_mask] = behavior_q_net.init_hidden(batch_size=len(next_hidden[done_mask])).to(self.device)
                next_target_hidden[done_mask] = behavior_q_net.init_hidden(batch_size=len(next_target_hidden[done_mask])).to(self.device)
                next_double_hidden[done_mask] = behavior_q_net.init_hidden(batch_size=len(next_double_hidden[done_mask])).to(self.device)
                
                hidden = next_hidden
                target_hidden = next_target_hidden
                double_hidden = next_double_hidden

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=behavior_q_net.parameters(), max_norm=self.grad_clip_norm, norm_type=2)
            optimizer.step()

            new_td_error = abs(target_values-sum_max_behavior_q)
            for index in range(self.batch_size):
                idx = priority_idx[index]
                Replay_buffer.update(idx, new_td_error[index])

