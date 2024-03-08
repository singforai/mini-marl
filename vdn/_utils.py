import os
import time
import torch
import random
import logging

import numpy as np


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def action_network(args, train_env):
    if args.action_network == "q_net":
        from _network import Q_Network
        target_network = Q_Network(observation_space=train_env.observation_space, action_space=train_env.action_space, args=args)
        behavior_network = Q_Network(observation_space=train_env.observation_space, action_space=train_env.action_space, args=args)
    elif args.action_network == "dueling_net":
        from _network import Dueling_Network
        target_network = Dueling_Network(observation_space=train_env.observation_space, action_space=train_env.action_space, args=args)
        behavior_network = Dueling_Network(observation_space=train_env.observation_space, action_space=train_env.action_space, args=args)
    else:
        raise Exception("Check a hyperparameter-action_network!")

    return target_network, behavior_network 

def target_setting(args):
    if args.target_setting == 'dqn':
        from _train import Train_dqn
        target_module = Train_dqn(args)
    elif args.target_setting == 'double_dqn':
        from _train import Train_double_dqn
        target_module = Train_double_dqn(args)
    else:
        raise Exception("Check a hyperparameter-target_setting!")
    return target_module

def cal_td_error(action, reward, done, behavior_q, target_q, gamma):
    action_index = action.long().reshape(-1, 1)
    behavior_value = torch.gather(input = behavior_q[0], dim=1, index = action_index).reshape(1, -1)[0].sum(dim=0)

    target_max_q = target_q.max(dim=2)[0][0].sum(dim=0)
    target_value = sum(reward)+(1-done)*gamma*target_max_q
    
    return abs(target_value-behavior_value).item()


def set_logging(experiment_name):
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    logging.basicConfig(filename=f'./logs/{experiment_name}-{int(time.time())}.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
def log_hyperparameter(args):

    logging.info("="*60)
    for hyperparameter, value in vars(args).items():
         logging.info(f"{hyperparameter:<30}| {value}")
    logging.info("="*60)

def chunk_initialize():
    state_chunk:      list = []
    action_chunk:     list = []
    reward_chunk:     list = []
    next_state_chunk: list = []
    done_chunk:       list = []
    td_error_chunk:   list = []
    return state_chunk, action_chunk, reward_chunk, next_state_chunk, done_chunk, td_error_chunk