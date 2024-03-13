import os
import time
import torch
import random
import logging

import numpy as np
import torch.nn as nn

from typing import Callable, Dict, Any


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def decide_network(args, train_env, device):
    from _network import Network, Q_Net, Dueling_Net
    
    networks: Dict[str, object] = {"q_net" : Q_Net, "dueling_net" : Dueling_Net}
    network_type: Callable[[str], object] = networks.get(args.decide_network, None)
    if network_type == None: raise Exception("Check a hyperparamete: network!")

    target_network = network_type(observation_space=train_env.observation_space, action_space=train_env.action_space, args=args).to(device)
    behavior_network = network_type(observation_space=train_env.observation_space, action_space=train_env.action_space, args=args).to(device)
    return target_network, behavior_network


def decide_target(Replay_buffer, behavior_network, target_network, args, device):
    from _train import Target_Dqn, Target_Double_Dqn

    train_targets: Dict[str, object] = {"dqn": Target_Dqn, "double_dqn": Target_Double_Dqn} 
    target_type: Callable[[str], object] = train_targets.get(args.decide_target, None) 
    if target_type == None: raise Exception("Check a hyperparameter: decide_target!")
    train_module = target_type(Replay_buffer,behavior_network, target_network, args, device)

    return train_module


def cal_td_error(action, reward, done, behavior_q, target_q, gamma):
    action_index = action.long().reshape(-1, 1)
    behavior_value = torch.gather(
        input=behavior_q[0], dim=1, index=action_index).reshape(1, -1)[0].sum(dim=0)

    target_max_q = target_q.max(dim=2)[0][0].sum(dim=0)
    target_value = sum(reward)+(1-done)*gamma*target_max_q

    return abs(target_value-behavior_value).item()


def set_logging(experiment_name: str):
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    logging.basicConfig(filename=f'./logs/{experiment_name}-{int(time.time())}.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def log_hyperparameter(args, device):

    logging.info("="*60)
    for hyperparameter, value in vars(args).items():
        logging.info(f"{hyperparameter:<30}| {value}")
    logging.info(f"Device: {device}")
    logging.info("="*60)
    


def chunk_initialize():
    state_chunk:      list = []
    action_chunk:     list = []
    reward_chunk:     list = []
    next_state_chunk: list = []
    done_chunk:       list = []
    td_error_chunk:   list = []
    return state_chunk, action_chunk, reward_chunk, next_state_chunk, done_chunk, td_error_chunk
