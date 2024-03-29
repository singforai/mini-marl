import numpy as np
import torch
from algorithms.qmix.algorithm.agent_q_function import AgentQFunction
from algorithms.base.recurrent_policy import RecurrentPolicy
from torch.distributions import Categorical, OneHotCategorical
from utils.util import get_dim_from_space, is_discrete, is_multidiscrete, make_onehot, DecayThenFlatSchedule, avail_choose, to_torch, to_numpy

import sys

class QMixPolicy(RecurrentPolicy):
    def __init__(self, args, config, policy_config, train=True):
        """
        QMIX/VDN Policy Class to compute Q-values and actions. See parent class for details.
        :param config: (dict) contains information about hyperparameters and algorithm configuration
        :param policy_config: (dict) contains information specific to the policy (obs dim, act dim, etc)
        :param train: (bool) whether the policy will be trained.
        """
        self.args = config["args"]
        self.device = config['device']
        self.num_agents = config["num_agents"]

        self.batch_size = args.batch_size

        self.obs_space = policy_config["obs_space"]
        self.obs_dim = get_dim_from_space(self.obs_space)
        self.act_space = policy_config["act_space"]
        self.act_dim = get_dim_from_space(self.act_space)
        self.output_dim = sum(self.act_dim) if isinstance(self.act_dim, np.ndarray) else self.act_dim
        self.hidden_size = self.args.hidden_size
        self.central_obs_dim = policy_config["cent_obs_dim"]
        self.discrete = is_discrete(self.act_space)
        self.multidiscrete = is_multidiscrete(self.act_space)

        #박현우 표 코드 변경
        device=torch.device("cuda:0")
        self.tpdv = dict(dtype=torch.float32, device=device)

        if self.args.prev_act_inp:
            # this is only local information so the agent can act decentralized
            self.q_network_input_dim = self.obs_dim + self.act_dim # 73 64 9
        else:
            self.q_network_input_dim = self.obs_dim

        # Local recurrent q network for the agent
        self.q_network = AgentQFunction(self.args, self.q_network_input_dim, self.act_dim, self.device)

        if train:
            self.exploration = DecayThenFlatSchedule(self.args.epsilon_start, self.args.epsilon_finish, self.args.epsilon_anneal_time,
                                                  decay="linear")

    def get_q_values(self, obs_batch, prev_action_batch, rnn_states, action_batch=None):
        """
        Computes q values using the given information.
        :param obs: (np.ndarray) agent observations from which to compute q values
        :param prev_actions: (np.ndarray) agent previous actions which are optionally an input to q network
        :param rnn_states: (np.ndarray) RNN states of q network
        :param action_batch: (np.ndarray) if not None, then only return the q values corresponding to actions in action_batch
        :return q_values: (torch.Tensor) computed q values
        :return new_rnn_states: (torch.Tensor) updated RNN states
        """

        """
        3x64 / 3x9 / 3x64

        61x96x64 / 61x96x9 / 96x64: runstate
        """

        if self.args.prev_act_inp: # 이전 action도 input으로 포함시키는가? => default: false
            obs_batch = to_torch(obs_batch).to(**self.tpdv)
            prev_action_batch = to_torch(prev_action_batch).to(**self.tpdv)
            input_batch = torch.cat((obs_batch, prev_action_batch), dim=-1) #3x73 / 61x96x73
        else:
            input_batch = obs_batch
        q_batch, new_rnn_states = self.q_network(input_batch, rnn_states) #collect: [3,9] #[3,64] => [61, 96, 9] , [96, 64]
        if action_batch is not None:
            action_batch = to_torch(action_batch).to(self.device)
            q_values = self.q_values_from_actions(q_batch, action_batch)
        else:
            q_values = q_batch
        return q_values, new_rnn_states

    def q_values_from_actions(self, q_batch, action_batch): # 60x96x9 , 60x96x9 // [61, 96, 9], [61,96,9]
        """
        Get q values corresponding to actions.
        :param q_batch: (torch.Tensor) q values corresponding to every action.
        :param action_batch: (torch.Tensor) actions taken by the agent.
        :return q_values: (torch.Tensor) q values in q_batch corresponding to actions in action_batch
        """

        if self.multidiscrete:
            ind = 0
            all_q_values = []
            for i in range(len(self.act_dim)):
                curr_q_batch = q_batch[i]
                curr_action_portion = action_batch[:, :, ind: ind + self.act_dim[i]]
                curr_action_inds = curr_action_portion.max(dim=-1)[1]
                curr_q_values = torch.gather(curr_q_batch, 2, curr_action_inds.unsqueeze(dim=-1))
                all_q_values.append(curr_q_values)
                ind += self.act_dim[i]
            q_values = torch.cat(all_q_values, dim=-1)
        else:
            # convert one-hot action batch to index tensors to gather the q values corresponding to the actions taken
            action_batch = action_batch.max(dim=-1)[1]#dim=-1은 가장 안쪽에 았는 차원을 기준으로 최대값 찾아냄 최대값과 인덱스를 같이 보여주나 여긴 [1] 이용해 index 계산 
            #action_batch: 60x91 or 60x90

            # import pdb; pdb.set_trace()
            q_values = torch.gather(q_batch, 2, action_batch.unsqueeze(dim=-1))  #action_batch: 맨 안쪽에 차원 1씩 추가 => 60x90x1 or 61x90x1
            #gather함수를 활용해 특정 index를 추출한다. q_values는 특정 action에 대한 q값이 된다. 

            # q_values is a column vector containing q values for the actions specified by action_batch
        return q_values
    
    # def transform_obs(self, obs):
    #     transformed_obs = np.array(obs).reshape(self.num_agents, -1)
    #     return transformed_obs

    def get_actions(self, obs, prev_actions, rnn_states, available_actions=None, t_env=None, explore=False):
        """See parent class."""
        q_values_out, new_rnn_states = self.get_q_values(obs, prev_actions, rnn_states) # input: (3, 64) (3, 9) (3, 64) / output: torch.Size([3, 9]) torch.Size([3, 64])
        onehot_actions, greedy_Qs = self.actions_from_q(q_values_out, available_actions=available_actions, explore=explore, t_env=t_env)
        return onehot_actions, new_rnn_states, greedy_Qs

    def actions_from_q(self, q_values, available_actions=None, explore=False, t_env=None):
        """
        Computes actions to take given q values.
        :param q_values: (torch.Tensor) agent observations from which to compute q values
        :param available_actions: (np.ndarray) actions available to take (None if all actions available)
        :param explore: (bool) whether to use eps-greedy exploration
        :param t_env: (int) env step at which this function was called; used to compute eps for eps-greedy
        :return onehot_actions: (np.ndarray) actions to take (onehot)
        :return greedy_Qs: (torch.Tensor) q values corresponding to greedy actions.
        """
        if self.multidiscrete:
            no_sequence = len(q_values[0].shape) == 2
            batch_size = q_values[0].shape[0] if no_sequence else q_values[0].shape[1]
            seq_len = None if no_sequence else q_values[0].shape[0]
        else:
            no_sequence = len(q_values.shape) == 2
            batch_size = q_values.shape[0] if no_sequence else q_values.shape[1]
            seq_len = None if no_sequence else q_values.shape[0]

        # mask the available actions by giving -inf q values to unavailable actions
        if available_actions is not None:
            q_values = q_values.clone()
            q_values = avail_choose(q_values, available_actions)
        else:
            q_values = q_values

        if self.multidiscrete:
            onehot_actions = []
            greedy_Qs = []
            for i in range(len(self.act_dim)):
                greedy_Q, greedy_action = q_values[i].max(dim=-1)

                if explore:
                    assert no_sequence, "Can only explore on non-sequences"
                    eps = self.exploration.eval(t_env)
                    rand_number = np.random.rand(batch_size)
                    # random actions sample uniformly from action space
                    random_action = Categorical(logits=torch.ones(batch_size, self.act_dim[i])).sample().numpy()
                    take_random = (rand_number < eps).astype(int)
                    action = (1 - take_random) * to_numpy(greedy_action) + take_random * random_action
                    onehot_action = make_onehot(action, self.act_dim[i])
                else:
                    greedy_Q = greedy_Q.unsqueeze(-1)
                    if no_sequence:
                        onehot_action = make_onehot(greedy_action, self.act_dim[i])
                    else:
                        onehot_action = make_onehot(greedy_action, self.act_dim[i], seq_len=seq_len)

                onehot_actions.append(onehot_action)
                greedy_Qs.append(greedy_Q)

            onehot_actions = np.concatenate(onehot_actions, axis=-1)
            greedy_Qs = torch.cat(greedy_Qs, dim=-1)
        else:
            greedy_Qs, greedy_actions = q_values.max(dim=-1)
            if explore:
                assert no_sequence, "Can only explore on non-sequences"
                eps = self.exploration.eval(t_env)
                rand_numbers = np.random.rand(batch_size)
                # random actions sample uniformly from action space
                logits = avail_choose(torch.ones(batch_size, self.act_dim), available_actions)
                random_actions = Categorical(logits=logits).sample().numpy()
                take_random = (rand_numbers < eps).astype(int)
                actions = (1 - take_random) * to_numpy(greedy_actions) + take_random * random_actions
                onehot_actions = make_onehot(actions, self.act_dim)
            else:
                greedy_Qs = greedy_Qs.unsqueeze(-1)
                if no_sequence:
                    onehot_actions = make_onehot(greedy_actions, self.act_dim)
                else:
                    onehot_actions = make_onehot(greedy_actions, self.act_dim, seq_len=seq_len)

        return onehot_actions, greedy_Qs

    def get_random_actions(self, obs, available_actions=None):
        """See parent class."""
        batch_size = obs.shape[0]
        if self.multidiscrete:
            random_actions = [OneHotCategorical(logits=torch.ones(batch_size, self.act_dim[i])).sample().numpy() for i in
                                range(len(self.act_dim))]
            random_actions = np.concatenate(random_actions, axis=-1)
        else:
            if available_actions is not None:
                logits = avail_choose(torch.ones(batch_size, self.act_dim), available_actions)
                random_actions = OneHotCategorical(logits=logits).sample().numpy()
            else:
                random_actions = OneHotCategorical(logits=torch.ones(batch_size, self.act_dim)).sample().numpy()
        
        return random_actions

    def init_hidden(self, num_agents, batch_size):
        """See parent class."""
        if num_agents == -1:
            return torch.zeros(batch_size, self.hidden_size)
        else:
            return torch.zeros(num_agents, batch_size, self.hidden_size)

    def parameters(self):
        """See parent class."""
        return self.q_network.parameters()

    def load_state(self, source_policy):
        """See parent class."""
        self.q_network.load_state_dict(source_policy.q_network.state_dict())