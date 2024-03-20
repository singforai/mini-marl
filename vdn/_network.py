import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, observation_space, action_space, args):
        super().__init__()
        self.use_recurrent: bool = args.use_recurrent
        self.num_agents: int = len(observation_space)

        self.observation_space = observation_space
        self.action_space = action_space

        self.feature_hidden: int = 64
        self.gru_input: int = 32
        self.gru_hidden: int = self.gru_input
        self.action_hidden: int = 32
        self.value_hidden: int = 32
        self.advantage_hidden: int = 32

        self.feature_net = nn.ParameterList()
        self.gru_net = nn.ParameterList()
        self.action_net = nn.ParameterList()
        self.value_net = nn.ParameterList()
        self.advantage_net = nn.ParameterList()

    def feature_network(self, agent_i: int = 0):
        n_obs: int = self.observation_space[agent_i].shape[0]
        self.feature_net.append(nn.Sequential(nn.Linear(n_obs, self.feature_hidden), nn.ReLU(),
                                              nn.Linear(self.feature_hidden, self.gru_input), nn.ReLU()))

    def gru_network(self):
        self.gru_net.append(nn.GRUCell(input_size=self.gru_input,
                                       hidden_size=self.gru_hidden))

    def action_network(self, agent_i: int = 0):
        self.action_net.append(nn.Sequential(nn.Linear(self.gru_hidden, self.action_space[agent_i].n)))

    def value_network(self):
        self.value_net.append(nn.Sequential(nn.Linear(self.gru_hidden, self.value_hidden),
                                            nn.Linear(self.value_hidden, 1)))

    def advantage_network(self, agent_i: int = 0):
        self.advantage_net.append(nn.Sequential(nn.Linear(self.gru_hidden, self.advantage_hidden),
                                                nn.Linear(self.advantage_hidden, self.action_space[agent_i].n)))

    def epsilon_greedy(self, q_tensor, epsilon):
        mask = (torch.rand(size=(q_tensor.shape[0],)) <= epsilon)
        action = torch.empty(size=(q_tensor.shape[0], q_tensor.shape[1]))
        action[mask] = torch.randint(
            low=0, high=q_tensor.shape[2], size=action[mask].shape).float()
        action[~mask] = q_tensor[~mask].argmax(dim=2).float()
        return action, q_tensor


class Q_Net(Network):
    def __init__(self, observation_space, action_space, args):
        super().__init__(observation_space, action_space, args)

        for agent_i in range(self.num_agents):
            super().feature_network(agent_i=agent_i)
            if self.use_recurrent:
                super().gru_network()
            super().action_network(agent_i=agent_i)

    def forward(self, obs, hidden):
        q_values: list = ([torch.empty(obs.shape[0])] * self.num_agents)
        next_hidden: list = [torch.empty(
            obs.shape[0], 1, self.gru_hidden)] * self.num_agents

        for agent_i in range(self.num_agents):
            feature = self.feature_net[agent_i](obs[:, agent_i, :])
            if self.use_recurrent:
                feature = self.gru_net[agent_i](feature, hidden[:, agent_i, :])
                next_hidden[agent_i] = feature.unsqueeze(1)
            q_values[agent_i] = self.action_net[agent_i](
                feature).unsqueeze(dim=1)
        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)

    def sample_action(self, obs, hidden, epsilon):
        q_tensor, next_hidden = self.forward(obs, hidden)
        action, q_tensor = super().epsilon_greedy(q_tensor, epsilon)
        return action, next_hidden, q_tensor

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.gru_hidden))

class Dueling_Net(Network):
    def __init__(self, observation_space, action_space, args):
        super().__init__(observation_space, action_space, args)
        for agent_i in range(self.num_agents):
            super().feature_network(agent_i=agent_i)
            if self.use_recurrent:
                super().gru_network()
            super().value_network()
            super().advantage_network(agent_i=agent_i)

    def forward(self, obs, hidden):
        advantage_values: list = (
            [torch.empty(obs.shape[0])] * self.num_agents)
        value_values: list = ([torch.empty(obs.shape[0])] * self.num_agents)
        action_values: list = ([torch.empty(obs.shape[0])] * self.num_agents)
        next_hidden: list = (
            [torch.empty(obs.shape[0], 1, self.gru_hidden)] * self.num_agents)

        for agent_i in range(self.num_agents):
            feature = self.feature_net[agent_i](obs[:, agent_i, :])
            if self.use_recurrent:
                feature = self.gru_net[agent_i](feature, hidden[:, agent_i, :])
                next_hidden[agent_i] = feature.unsqueeze(1)

            advantage_values[agent_i] = self.advantage_net[agent_i](
                feature).unsqueeze(dim=1)
            value_values[agent_i] = self.value_net[agent_i](
                feature).unsqueeze(dim=0)

            action_values[agent_i] = value_values[agent_i] + \
                (advantage_values[agent_i] -
                 advantage_values[agent_i].mean(dim=2))

        return torch.cat(action_values, dim=1), torch.cat(next_hidden, dim=1)

    def sample_action(self, obs, hidden, epsilon):
        q_tensor, next_hidden = self.forward(obs, hidden)
        action, q_tensor = super().epsilon_greedy(q_tensor, epsilon)
        return action, next_hidden, q_tensor

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.gru_hidden))
