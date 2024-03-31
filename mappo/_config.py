import argparse


def get_config() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Proximal Policy Optimization (PPO)")
    param = parser.add_argument

    # wandb setting
    param("--use_wandb", action='store_true', default=False, help="Whether to use wandb")
    param("--entity_name", type=str, default="singfor7012", help="wandb_name")
    param(
        "--env_name",
        type=str,
        default="ma_gym:Checkers-v0",
        help="Built-in Env for ma_gym",
    )
    param(
        "--experiment_name",
        type=str,
        default="ppo",
        help="Experiment title stored in Wandb",
    )
    param(
        "--group_name",
        type = str,
        default="None",
        help="Experiment group title stored in Wandb",
    )

    # cuda setting
    param(
        "--use_cuda",
        type=bool,
        default=True,
        help="Decide whether to use GPU during training ",
    )
    param(
        "--n_training_threads",
        type=int,
        default=12,
        help="Number of threads to use for CPU internal calculations",
    )  

    # seed setting
    param(
        "--fix_seed",
        type=bool,
        default=True,
        help="Should I fix the seed during training?",
    )  # epsilon control parameters
    param("--seed", type=int, default=42, help="Choose training seed")

    # env setting
    param("--max_episodes",
           type=int, 
           default=100000, 
           help="Number of episodes trained"
    )
    param("--max_step", type=int, default=100, help="Maximum support step per episode")
    param("--step_cost", type=float, default=-0.01, help="Rewards given per step")

    # render setting 
    param("--use_render", type=bool, default=False, help="Render the learning process")
    param("--sleep_second", type=float, default=0.0, help=" Runtime of time.sleep")

    # batch setting
    param("--sampling_batch_size", 
          type=int, 
          default=1, 
          help="batch size of data when the model samples the episode")
    param(
        "--training_batch_size",
        type=int,
        default=1,
        help="batch size of data when training PPO",
    )
    param(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Number of parallel envs for evaluating rollouts",
    )

    # gae setting 
    param(
        "--use_gae",
        action="store_false",
        default=True,
        help="use generalized advantage estimation",
    )
    param(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="The closer to 1, the smaller the bias of the predicted value and the larger the variance.",
    )

    # gradient norm setting
    param(
        "--use_max_grad_norm",
        action="store_false",
        default=True,
        help="by default, use max norm of gradients. If set, do not use.",
    )
    param(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="max norm of gradients (default: 0.5)",
    )



















    ## train parameters
    param(
        "--algorithm_name",
        type=str,
        default="rmappo",
        choices=["rmappo", "mappo", "ippo"],
        help="Name of algorithm to apply",
    )
    param(
        "--share_policy",
        action='store_false',
        default=True,  
        help="Determining if agents want to share the same network with same parameters",
    )
    param(
    "--use_common_reward",
    action = 'store_true',
    default=False,  
    help="Each agent will decide whether to receive the sum of rewards from all agents or to receive rewards separately for each agent.",
    )
    param(
        "--use_centralized_V",
        action = "store_true",
        default=False, 
        help="Whether to use centralized V function",
    )
    param(
        "--use_linear_lr_decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )
    param(
        "--log_interval",
        type=int,
        default=10,
        help="time duration between contiunous twice log printing.",
    )
    param(
        "--use_eval",
        action="store_true",
        default=True,
        help="by default, do not start evaluation. If set`, start evaluation alongside with training.",
    )
    param(
        "--save_interval",
        type=int,
        default=1,
        help="time duration between contiunous twice models saving.",
    )
    param(
        "--use_feature_normalization",
        action="store_false",
        default=True,
        help="Whether to apply layernorm to the inputs",
    )
    param("--use_ReLU", action="store_false", default=True, help="Whether to use ReLU")
    param(
        "--stacked_frames",
        type=int,
        default=1,
        help="Dimension of hidden layers for actor/critic networks",
    )
    param(
        "--layer_N",
        type=int,
        default=1,
        help="Number of layers for actor/critic networks",
    )

    param(
        '--use_mix_advantage',
        action = "store_true",
        default = False,
        help = "When calculating the advantage value, decide whether to average the calculated value of each agent.",
    )

    # Common
    param(
        "--hidden_size",
        type=int,
        default=32,
        help="Dimension of hidden layers for actor/critic networks",
    )
    param(
        "--use_popart",
        action="store_true",
        default=False,
        help="by default False, use PopArt to normalize rewards.",
    )
    param("--recurrent_N", type=int, default=1, help="The number of recurrent layers")
    param(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor used to calculate TD error",
    )


    # optimizer hyperparameter
    param(
        "--actor_lr", type=float, default=1e-4, help="Learning rate of optimizer Adam"
    )
    param("--critic_lr", type=float, default=1e-4, help="critic learning rate")
    param("--opti_eps", type=float, default=1e-5, help="RMSprop optimizer epsilon")
    param(
        "--weight_decay",
        type=float,
        default=0,
        help="weight decay (L2 penalty): A penalty proportional to weight decay is given.",
    )

    # layer initialize
    param(
        "--gain",
        type=float,
        default=0.01,
        help="The gain # of last action layer, weight scaling",
    )
    param(
        "--use_orthogonal",
        action="store_false",
        default=True,
        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases",
    )

    # network hyperparameter
    param(
        "--use_recurrent_policy",
        action="store_false",
        default=True,
        help="use a recurrent policy",
    )
    param(
        "--use_naive_recurrent_policy",
        action="store_true",
        default=False,
        help="Whether to use a naive recurrent policy",
    )

    param(
        "--use_clipped_value_loss",
        action="store_false",
        default=True,
        help="by default, clip loss value. If set, do not clip loss value.",
    )
    param(
        "--use_huber_loss",
        action="store_false",
        default=True,
        help="by default, use huber loss. If not, do not use huber loss.",
    )
    param(
        "--use_policy_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in policy loss.",
    )
    param(
        "--use_value_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in value loss.",
    )
    param(
        "--use_valuenorm",
        action="store_false",
        default=True,
        help="by default True, use running mean and std to normalize rewards.",
    )

    param(
        "--clip_param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    param(
        "--ppo_epoch", type=int, default=15, help="number of ppo epochs (default: 15)"
    )

    param(
        "--data_chunk_length",
        type=int,
        default=5,
        help="Time length of chunks used to train a recurrent_policy",
    )
    param(
        "--value_loss_coef",
        type=float,
        default=0.5,
        help="value loss coefficient (default: 0.5)",
    )
    param(
        "--entropy_coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )

    param(
        "--huber_delta", type=float, default=10.0, help=" coefficience of huber loss."
    )

    # buffer hyperparameter


    #  eval
    param("--eval_interval", type=float, default=1, help="evaluation interval")
    return parser
