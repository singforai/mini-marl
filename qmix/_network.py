import torch
import torch.nn as nn


class Q_Net(nn.Module):
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
            feature_output = getattr(self, f'feature_network_{agent_i}')(obs[:, agent_i, :])
            if self.use_recurrent:
                feature_output = getattr(self, f'gru_network_{agent_i}')(
                    feature_output, hidden[:, agent_i, :])  # gru_output: 1x32 or 32x32
                next_hidden[agent_i] = feature_output.unsqueeze(1)  # 1x1x32

            q_values[agent_i] = getattr(self, f'action_network_{agent_i}')(
                feature_output).unsqueeze(dim=1)
            
        return torch.cat(q_values, dim=1).to("cpu"), torch.cat(next_hidden, dim=1)

    def sample_action(self, obs, hidden, epsilon):
        q_tensor, next_hidden = self.forward(obs, hidden)
        mask = (torch.rand(size=(q_tensor.shape[0],)) <= epsilon)
        action = torch.empty(size=(q_tensor.shape[0], q_tensor.shape[1]))
        action[mask] = torch.randint(low=0, high=q_tensor.shape[2], size=action[mask].shape).float()
        action[~mask] = q_tensor[~mask].argmax(dim=2).float()
        return action, next_hidden, q_tensor

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.gru_hidden_size))
    
class Dueling_Net(nn.Module):
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

        return torch.cat(action_values, dim=1).to("cpu"), torch.cat(next_hidden, dim=1)

    def sample_action(self, obs, hidden, epsilon):
        q_tensor, next_hidden = self.forward(obs, hidden)
        mask = (torch.rand(size=(q_tensor.shape[0],)) <= epsilon)
        action = torch.empty(size=(q_tensor.shape[0], q_tensor.shape[1]))
        action[mask] = torch.randint(low=0, high=q_tensor.shape[2], size=action[mask].shape).float()
        action[~mask] = q_tensor[~mask].argmax(dim=2).float()
        return action, next_hidden, q_tensor

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.gru_hidden_size))

class Mix_Net(nn.Module):
    def __init__(self, observation_space, args):
        super().__init__()
        hidden_dim=32
        hx_size=64
        state_size = sum([_.shape[0] for _ in observation_space])
        self.hidden_dim = hidden_dim
        self.hx_size = hx_size
        self.n_agents = len(observation_space)
        self.recurrent = args.use_recurrent

        hyper_net_input_size = state_size
        if self.recurrent:
            self.gru = nn.GRUCell(state_size, self.hx_size)
            hyper_net_input_size = self.hx_size
        self.hyper_net_weight_1 = nn.Linear(hyper_net_input_size, self.n_agents * hidden_dim)
        self.hyper_net_weight_2 = nn.Linear(hyper_net_input_size, hidden_dim)

        self.hyper_net_bias_1 = nn.Linear(hyper_net_input_size, hidden_dim)
        self.hyper_net_bias_2 = nn.Sequential(nn.Linear(hyper_net_input_size, hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(hidden_dim, 1))

    def forward(self, q_values, observations, hidden):
        batch_size, n_agents, obs_size = observations.shape
        state = observations.view(batch_size, n_agents * obs_size)

        x = state
        if self.recurrent:
            hidden = self.gru(x, hidden)
            x = hidden

        weight_1 = torch.abs(self.hyper_net_weight_1(x))
        weight_1 = weight_1.view(batch_size, self.hidden_dim, n_agents)
        bias_1 = self.hyper_net_bias_1(x).unsqueeze(-1)
        weight_2 = torch.abs(self.hyper_net_weight_2(x))
        bias_2 = self.hyper_net_bias_2(x)

        x = torch.bmm(weight_1, q_values.unsqueeze(-1)) + bias_1
        x = torch.relu(x)
        x = (weight_2.unsqueeze(-1) * x).sum(dim=1) + bias_2
        return x, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.hx_size))