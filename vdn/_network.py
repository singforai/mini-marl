import torch
import torch.nn as nn


class Q_Network(nn.Module):
    def __init__(self, observation_space, action_space, args):
        super().__init__()
        self.use_recurrent: bool = args.use_recurrent

        self.num_agents: int = len(observation_space)
        self.feature_hidden1_size: int = 64
        self.gru_input_size: int = 32
        self.gru_hidden_size: int = 32

        for agent_i in range(self.num_agents):
            n_obs: int = observation_space[agent_i].shape[0]
            setattr(self, f'feature_network_{agent_i}', nn.Sequential(nn.Linear(n_obs, self.feature_hidden1_size), nn.ReLU(
            ), nn.Linear(self.feature_hidden1_size, self.gru_input_size), nn.ReLU()))
            setattr(self, f'gru_network_{agent_i}', nn.GRUCell(
                input_size=self.gru_input_size, hidden_size=self.gru_hidden_size)) if self.use_recurrent else None
            setattr(self, f'action_network_{agent_i}', nn.Sequential(nn.Linear(
                self.gru_hidden_size, action_space[agent_i].n)))

    def forward(self, obs, hidden):
        q_values: list = ([torch.empty(obs.shape[0])] * self.num_agents)
        next_hidden: list = [torch.empty(
            obs.shape[0], 1, self.gru_hidden_size)] * self.num_agents

        for agent_i in range(self.num_agents):
            feature_output = getattr(
                self, f'feature_network_{agent_i}')(obs[:, agent_i, :])
            if self.use_recurrent:
                feature_output = getattr(self, f'gru_network_{agent_i}')(
                    feature_output, hidden[:, agent_i, :])  # gru_output: 1x32 or 32x32
                next_hidden[agent_i] = feature_output.unsqueeze(1)  # 1x1x32

            q_values[agent_i] = getattr(self, f'action_network_{agent_i}')(
                feature_output).unsqueeze(dim=1)
            
        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)

    def sample_action(self, obs, hidden, epsilon):
        q_tensor, next_hidden = self.forward(obs, hidden)
        mask = (torch.rand(size=(q_tensor.shape[0],)) <= epsilon)
        action = torch.empty(size=(q_tensor.shape[0], q_tensor.shape[1]))
        action[mask] = torch.randint(low=0, high=q_tensor.shape[2], size=action[mask].shape).float()
        action[~mask] = q_tensor[~mask].argmax(dim=2).float()
        return action, next_hidden, q_tensor

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.gru_hidden_size))
    
class Dueling_Network(nn.Module):
    def __init__(self, observation_space, action_space, args):
        super().__init__()
        self.use_recurrent: bool = args.use_recurrent

        self.num_agents: int = len(observation_space)
        self.feature_hidden1_size: int = 64
        self.gru_input_size: int = 32
        self.gru_hidden_size: int = 32

        for agent_i in range(self.num_agents):
            n_obs: int = observation_space[agent_i].shape[0]
            setattr(self, f'feature_network_{agent_i}', nn.Sequential(nn.Linear(n_obs, self.feature_hidden1_size), nn.ReLU(), nn.Linear(self.feature_hidden1_size, self.gru_input_size), nn.ReLU()))
            setattr(self, f'gru_network_{agent_i}', nn.GRUCell(input_size=self.gru_input_size, hidden_size=self.gru_hidden_size)) if self.use_recurrent else None
            
            setattr(self, f'advantage_network_{agent_i}', nn.Sequential(nn.Linear(self.gru_hidden_size, action_space[agent_i].n)))
            setattr(self, f'value_network_{agent_i}', nn.Sequential(nn.Linear(self.gru_hidden_size, 1)))

    def forward(self, obs, hidden):
        advantage_values: list = ([torch.empty(obs.shape[0])] * self.num_agents)
        value_values: list = ([torch.empty(obs.shape[0])] * self.num_agents)
        action_values: list = ([torch.empty(obs.shape[0])] * self.num_agents)

        next_hidden: list = [torch.empty(
            obs.shape[0], 1, self.gru_hidden_size)] * self.num_agents

        for agent_i in range(self.num_agents):
            feature_output = getattr(
                self, f'feature_network_{agent_i}')(obs[:, agent_i, :])
            if self.use_recurrent:
                feature_output = getattr(self, f'gru_network_{agent_i}')(
                    feature_output, hidden[:, agent_i, :])  # gru_output: 1x32 or 32x32
                next_hidden[agent_i] = feature_output.unsqueeze(1)  # 1x1x32

            advantage_values[agent_i] = getattr(self, f'advantage_network_{agent_i}')(feature_output).unsqueeze(dim=1)
            value_values[agent_i] = getattr(self, f'value_network_{agent_i}')(feature_output).unsqueeze(dim=0)

            action_values[agent_i] = value_values[agent_i] + (advantage_values[agent_i]-torch.mean(input = advantage_values[agent_i], dim = 2))

        return torch.cat(action_values, dim=1), torch.cat(next_hidden, dim=1)

    def sample_action(self, obs, hidden, epsilon):
        q_tensor, next_hidden = self.forward(obs, hidden)
        mask = (torch.rand(size=(q_tensor.shape[0],)) <= epsilon)
        action = torch.empty(size=(q_tensor.shape[0], q_tensor.shape[1]))
        action[mask] = torch.randint(low=0, high=q_tensor.shape[2], size=action[mask].shape).float()
        action[~mask] = q_tensor[~mask].argmax(dim=2).float()
        return action, next_hidden, q_tensor

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.gru_hidden_size))

