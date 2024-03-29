import os
import time
import torch
import random
import logging

import numpy as np

def fix_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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