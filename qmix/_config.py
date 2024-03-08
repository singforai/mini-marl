import argparse


def get_config() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Qmix')
    param = parser.add_argument

    # wandb setting
    param('--use_wandb', type=bool, default=False, help="Whether to use wandb")
    param('--entity_name', type=str, default='singfor7012', help = "wandb_name")

    # experiment base setting parameters
    param('--env_name', type=str, default='ma_gym:Checkers-v0', help="Built-in environment settings for ma_gym")
    param('--experiment_name', type=str, default='vdn', help="Experiment title stored in Wandb")
    
    param('--use_cuda', type=bool, default=True, help="Decide whether to use GPU during training ")
    param('--n_training_threads', type=int, default=12, help="Number of threads to use for CPU internal calculations")


    # rendering parameters
    param('--use_render', type=bool, default=False,
          help="Render the learning process")
    param('--use_time_sleep', type=bool, default=False,
          help="Adjust rendering speed by stopping sleep_second with program set at each step")
    param('--sleep_second', type=float, default=0.5,
          help=" Runtime of use_time_sleep")

    # seed parameters
    param('--seed', type=int, default=42, help="Choose training seed")
    param('--fix_seed', type=bool, default=True,
          help="Should I fix the seed during training?")

    # epsilon control parameters
    param('--epsilon_anneal_episode', type=int, default=60000,
          help="Episode where epsilon starts to reach minimum")
    param('--max_epsilon', type=float, default=0.9,
          help="Epsilon set at the beginning of learning")
    param('--min_epsilon', type=float, default=0.05,
          help="Epsilon applied after epsilon_anneal_episode")

    # train model setting parameters
    param('--lr', type=float, default=1e-3,
          help="Learning rate of optimizer Adam")
    param('--gamma', type=float, default=0.99,
          help="Discount factor used to calculate TD error")
    param('--batch_size', type=int, default=32,
          help="Number of samples used for one training")
    param('--max_episodes', type=int, default=100000,
          help="Number of episodes trained")
    param('--max_step', type=int, default=100,
          help="Maximum support step per episode")
    param('--step_cost', type=float, default=-
          0.01, help="Rewards given per step")
    param('--chunk_size', type=int, default=10,
          help="Number of past steps input(trajectory): It must be a divisor of max_episode")
    param('--update_iter', type=int, default=10,
          help="Decide how many times to train on sampled data")
    param('--update_target_interval', type=int, default=20,
          help="Episode interval at which the target network copies the parameters of the behavioral network")
    param('--grad_clip_norm', type=int, default=5,
          help="Limit the maximum value of gradient's L2 norm")

    # Architecture parameters
    param('--target_setting', type = str, default = "dqn", 
          help="Determine how to set target value during training [option: dqn/double_dqn]")
    param('--action_network', type = str, default = "q_net", 
          help="Determine the form of the network to approximate the value of action [option: q_net/dueling_net]") 
    param('--use_recurrent', type=bool, default=True,
          help="Determine whether to use GRU for training")


    # test parameters
    param('--test_interval', type=int, default=10,
          help="Intervals at which tests are performed and results are displayed during training")
    param('--test_episodes', type=int, default=10,
          help="Number of tested episodes")

    # Replay_buffer parameters
    param('--buffer_limit', type=int, default=50000,
          help="It determines the capacity of Replay buffer and warm up chunk step")
    param('--eps', type=int, default=1e-6,
          help="small positive constant that reacts when TD-error is zero")
    param('--alpha', type=float, default=0.8,
          help="How much priority to reflect(alpha=0: uniform distribution ~ alpha=1: prioritized_distribution")
    param('--beta', type=float, default=0.2,
          help="beta=0: Not use importance sampling ~ beta=1: fully compensate importance sampling correction")
    param('--update_alpha_beta', type=bool, default=True,
          help="Update alpha and beta values linearly")

    return parser
