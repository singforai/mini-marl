import  gym
import torch
import numpy as np
import torch.nn as nn
from utils.util import update_linear_schedule
from algorithms.r_actor_critic import R_Actor

class R_MAPPO_Actor_Policy(nn.Module):
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, act_space, num_agents, device):
        super().__init__()
        self.device: torch.device = device

        self.obs_space = obs_space
        self.act_space = act_space
        self.opti_eps: float = args.opti_eps
        self.actor_lr: float = args.actor_lr
        self.weight_decay: float = args.weight_decay
        
        self.actor = nn.ModuleList()

        for agent_id in range(num_agents):
            self.actor.append(
                R_Actor(
                    args = args, 
                    obs_space = self.obs_space[agent_id], 
                    action_space = self.act_space[agent_id], 
                    device = self.device
                )
            )

        self.actor_optimizer = torch.optim.Adam(
            params=self.actor.parameters(), 
            lr=self.actor_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )


    def act(self, obs, rnn_states_actor, masks, actor, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, rnn_states_actor