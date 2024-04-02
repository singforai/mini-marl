import  gym
import torch
import numpy as np
from utils.util import update_linear_schedule
from algorithms.r_actor_critic import R_Actor

class R_MAPPO_Actor_Policy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, act_space, device):

        self.device: torch.device = device

        # optimizer hyperparameters
        self.actor_lr: float = args.actor_lr
        self.opti_eps: float = args.opti_eps
        self.weight_decay: float = args.weight_decay

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )