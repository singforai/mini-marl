import gym
import sys
import time
import torch

from utils.main_utils import *
from _config import get_config


def main(args):
    parser = get_config()
    args = parser.parse_known_args(args)[0]

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

    if args.use_cuda and torch.cuda.is_available():
        torch.set_num_threads(args.n_training_threads)
        device = torch.device(f"cuda:{args.num_gpu}")
    else:
        torch.set_num_threads(args.n_training_threads)
        device = torch.device("cpu")

    # set_logging(experiment_name=args.experiment_name)
    # log_hyperparameter(args=args, device=device)

    fix_random_seed(seed = args.seed) if args.fix_seed else None

    env = gym.make(
        id=args.env_name,
        full_observable=False,
        max_steps=args.max_step,
        step_cost=args.step_cost,
    )

    train_env_batch = [env for _ in range(args.sampling_batch_size)]

    config = {
        "args": args,
        "train_env": train_env_batch,
        "eval_env": env,
        "num_agents": env.n_agents,
        "device": device,
    }

    if args.algorithm_name == "rmappo":
        if args.use_recurrent_policy ^ args.use_naive_recurrent_policy:
            print(f"You are choosing to use rmappo!")
        else:
            raise Exception("Either use_recurrent_policy or use_naive_recurrent_policy must be set to True.")

    elif args.algorithm_name == "mappo":
        print("You are choosing to use mappo, we have to set use_recurrent_policy & use_naive_recurrent_policy to be False")
        args.use_recurrent_policy = False
        args.use_naive_recurrent_policy = False

    elif args.algorithm_name == "ippo":
        print("You are choosing to use ippo, we have to set use_centralized_V to be False")
        args.use_centralized_V = False
    else:
        raise Exception("Check the algorithm_name!")

    if args.policy_type == "share":
        from runner.shared.magym_runner import MAGYM_Runner as Runner
    elif args.policy_type == "separate":
        from runner.separated.magym_runner import MAGYM_Runner as Runner
    elif args.policy_type == "hybrid":
        from runner.hybrid.magym_runner import MAGYM_Runner as Runner
    else:
        raise Exception("Check the hyperparameter: policy_type!")

    runner = Runner(config)
    runner.run()
    
    env.close()
    run_wandb.finish()

if __name__ == "__main__":
    main(args = sys.argv[1:])
