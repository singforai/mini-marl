import os
import gym
import sys
import time
import torch

from tqdm import tqdm

from utils.main_utils import *

from config import get_config 

def main(args):
    parser = get_config()
    args = parser.parse_known_args(args, parser)[0]

    assert args.n_rollout_threads == 1, ("only support 1 env in recurrent version.") 

    if args.use_cuda and torch.cuda.is_available():
        torch.set_num_threads(args.n_training_threads)
        device = torch.device("cuda")
    else:
        torch.set_num_threads(args.n_training_threads) 
        device = torch.device("cpu")

    if args.use_wandb:
        import wandb
        project_name = args.env_name.split(":")[1] 
        run_wandb = wandb.init(
            entity=args.entity_name,
            project=project_name,
            group = args.group_name,
            name=f"{args.experiment_name}-{int(time.time())}",
            config=args,
            reinit=True,
        )
    
    # set_logging(experiment_name=args.experiment_name)
    # log_hyperparameter(args=args, device=device)

    fix_random_seed(seed = args.seed) if args.fix_seed else None

    env = gym.make(
        id=args.env_name,
        full_observable=False,\
        max_steps=args.episode_length,
        step_cost=args.step_cost,
    )

    config = {"args": args,
              "train_env": env,
              "eval_env": env,
              "num_agents": env.n_agents,
              "device": device}

    if args.share_policy:
        from runner.shared.magym_runner import MAGYM_Runner as Runner
    else:
        NotImplementedError
    
    total_num_steps: int = 0

    runner = Runner(config=config) 
    pbar = tqdm(total=args.num_env_steps, desc="training", ncols=70)
    while total_num_steps < args.num_env_steps:
        total_num_steps = runner.run() #base_runner에서 run함수를 호출
        pbar.update(total_num_steps)

    env.close()
    if args.use_wandb:
        run_wandb.finish()

if __name__ == "__main__":
    main(args = sys.argv[1:]) 
