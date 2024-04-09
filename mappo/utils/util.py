import numpy as np
import math
import torch
from typing import List

import os
import time
import torch
import random
import logging


def split_chunks_into_n_ags(args, data):
    n_r_t, n_agents, ep_len = args.training_batch_size, args.n_agents, args.max_step
    ch_len = args.data_chunk_length
    mini_b_size = n_r_t * n_agents * ep_len // ch_len
    # (n_r_t*ep_len*n_ags, XX_dim) -> (n_r_t, ep_len, n_ags, XX_dim)
    data = data.reshape(ch_len, -1, data.shape[-1])
    data_ags = data.permute(1, 0, 2)
    data_ags = torch.cat([data_ags[m] for m in range(mini_b_size)], dim=0)
    data_ags = data_ags.reshape(n_r_t, n_agents, ep_len, -1).permute(0, 2, 1, 3)
    return data_ags


def fix_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(experiment_name: str):
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    logging.basicConfig(
        filename=f"./logs/{experiment_name}-{int(time.time())}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def log_hyperparameter(args, device):

    logging.info("=" * 60)
    for hyperparameter, value in vars(args).items():
        logging.info(f"{hyperparameter:<30}| {value}")
    logging.info(f"Device: {device}")
    logging.info("=" * 60)


def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)


def get_grad_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e**2 / 2


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


def obs_sharing(obs: List, num_agents: int) -> np.array:
    share_obs_list = [np.array(obs).reshape(-1) for _ in range(num_agents)]
    return share_obs_list


def convert_each_rewards(rewards_batch):
    converted_rewards = [[[reward] for reward in rewards] for rewards in rewards_batch]
    return converted_rewards


def convert_sum_rewards(rewards_batch):
    converted_rewards = [[[sum(rewards)] for _ in rewards] for rewards in rewards_batch]
    return converted_rewards
