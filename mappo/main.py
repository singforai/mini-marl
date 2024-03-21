import pdb
import gym
import sys
import time
import torch
import logging
import torch.optim as optim

from tqdm import tqdm

from _utils import *
from _config import get_config


def main(args):
    parser = get_config()
    args = parser.parse_known_args(args)[0]

    if args.algorithm_name == "rmappo":
        print("You are choosing to use rmappo, we set use_recurrent_policy to be True")
        args.use_recurrent_policy = True
        args.use_naive_recurrent_policy = False
    elif args.algorithm_name == "mappo":
        print(
            "You are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False"
        )
        args.use_recurrent_policy = False
        args.use_naive_recurrent_policy = False
    elif args.algorithm_name == "ippo":
        print("You are choosing to use ippo, we set use_centralized_V to be False")
        args.use_centralized_V = False
    else:
        raise NotImplementedError

    if args.use_wandb:
        import wandb

        wandb.init(
            entity=args.entity_name,
            project=args.project_name,
            name=f"{args.experiment_name}-{int(time.time())}",
            config=args,
            reinit=True,
        )

    if args.use_cuda and torch.cuda.is_available():
        torch.set_num_threads(args.n_training_threads)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # set_logging(experiment_name=args.experiment_name)
    # log_hyperparameter(args=args, device=device)

    fix_random_seed(args.seed) if args.fix_seed else None

    train_env = gym.make(
        id=args.env_name,
        full_observable=False,
        max_steps=args.max_step,
        step_cost=args.step_cost,
    )
    eval_env = gym.make(
        id=args.env_name, max_steps=args.max_step, step_cost=args.step_cost
    )

    config = {
        "args": args,
        "train_env": train_env,
        "eval_env": eval_env,
        "num_agents": train_env.n_agents,
        "device": device,
    }

    if args.share_policy:
        from runner.shared.magym_runner import MAGYM_Runner as Runner
    else:
        from runner.seperated.magym_runner import MAGYM_Runner as Runner

    runner = Runner(config)
    runner.run()

    train_env.close()
    if args.use_eval and eval_env is not train_env:
        eval_env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
