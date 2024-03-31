import argparse


def get_config():
    parser = argparse.ArgumentParser(description="OFF-POLICY")
    param = parser.add_argument

    # prepare parameters
    param("--use_wandb", action='store_true', default=False, help="Whether to use wandb")
    param("--entity_name", type=str, default="singfor7012", help="wandb_name")
    param("--use_cuda", action='store_true', default=True, help="True uses GPU")
    param("--fix_seed", action='store_false', default=True, help="If True, all random seeds are fixed.")
    param("--step_cost", type = bool, default = -0.01, help="Cost(rewards) given at each step.")
    param(
        "--group_name",
        default=None,
        help="Experiment group title stored in Wandb",
    )

    param("--algorithm_name", type=str, default="vdn", choices=["rmatd3", "rmaddpg", "rmasac", "qmix", "vdn"])
    param("--experiment_name", type=str, default="vdn")
    param("--seed", type=int, default=1,
                        help="Random seed for numpy/torch")
    param("--cuda", action='store_false', default=True)
    param("--cuda_deterministic",
                        action='store_false', default=True)
    param('--n_training_threads', type=int,
                        default=1, help="Number of torch threads for training")
    param('--n_rollout_threads', type=int,  default=1,
                        help="Number of parallel envs for training rollout")
    param('--n_eval_rollout_threads', type=int,  default=1,
                        help="Number of parallel envs for evaluating rollout")
    param(
    "--use_common_reward",
    action = 'store_true',
    default=False,  
    help="Each agent will decide whether to receive the sum of rewards from all agents or to receive rewards separately for each agent.",
    )
    # env parameters
    param(
        "--env_name",
        type=str,
        default="ma_gym:Checkers-v0",
        help="Built-in settings for ma_gym",
    )
    param("--max_episodes", type=int, default=100000, help="Number of episodes trained")

    # replay buffer parameters
    param('--episode_length', type=int,
                        default=100, help="Max length for any episode")
    param('--buffer_size', type=int, default=10000,
                        help="Max # of transitions that replay buffer can contain")
    param('--use_reward_normalization', action='store_true',
                        default=False, help="Whether to normalize rewards in replay buffer")
    param('--use_popart', action='store_true', default=False,
                        help="Whether to use popart to normalize the target loss")
    param('--popart_update_interval_step', type=int, default=2,
                        help="After how many train steps popart should be updated")
                        
    # prioritized experience replay
    param('--use_per', action='store_false', default=True,
                        help="Whether to use prioritized experience replay") # 우선순위 경험 재생
    param('--per_nu', type=float, default=0.9,
                        help="Weight of max TD error in formation of PER weights")
    param('--per_alpha', type=float, default=0.6,
                        help="Alpha term for prioritized experience replay")
    param('--per_eps', type=float, default=1e-6,
                        help="Eps term for prioritized experience replay") # 우선순위를 매길 때 사용하는 작은 값
    param('--per_beta_start', type=float, default=0.4,
                        help="Starting beta term for prioritized experience replay")

    # network parameters
    param("--use_centralized_Q", action='store_false', 
                        default=True, help="Whether to use centralized Q function")
    param('--share_policy', action='store_false',
                        default=True, help="Whether agents share the same policy")
    param('--hidden_size', type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks")
    param('--layer_N', type=int, default=1,
                        help="Number of layers for actor/critic networks")
    param('--use_ReLU', action='store_false',
                        default=True, help="Whether to use ReLU")
    param('--use_feature_normalization', action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    param('--use_orthogonal', action='store_false', default=True, #Xavier와 다른 가충치 초기화 방식: False일 경우 
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    param("--gain", type=float, default=0.01, #qmix에서 사용
                        help="The gain # of last action layer")
    param("--use_conv1d", action='store_true',
                        default=False, help="Whether to use conv1d")
    param("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")

    # recurrent parameters
    param('--prev_act_inp', action='store_true', default=False,
                        help="Whether the actor input takes in previous actions as part of its input")
    param("--use_rnn_layer", action='store_false',
                        default=True, help='Whether to use a recurrent policy')
    param("--use_naive_recurrent_policy", action='store_false',
                        default=True, help='Whether to use a naive recurrent policy')
    # TODO now only 1 is support
    param("--recurrent_N", type=int, default=1)
    param('--data_chunk_length', type=int, default=5,
                        help="Time length of chunks used to train via BPTT")
    param('--burn_in_time', type=int, default=0,
                        help="Length of burn in time for RNN training, see R2D2 paper")

    # attn parameters
    param("--attn", action='store_true', default=False)
    param("--attn_N", type=int, default=1)
    param("--attn_size", type=int, default=64)
    param("--attn_heads", type=int, default=4)
    param("--dropout", type=float, default=0.0)
    param("--use_average_pool",
                        action='store_false', default=True)
    param("--use_cat_self", action='store_false', default=True)

    # optimizer parameters
    param('--lr', type=float, default=5e-4,
                        help="Learning rate for Adam")
    param("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    param("--weight_decay", type=float, default=0)

    # algo common parameters
    param('--batch_size', type=int, default=32,
                        help="Number of buffer transitions to train on at once")
    param('--gamma', type=float, default=0.99,
                        help="Discount factor for env")
    param("--use_max_grad_norm",
                        action='store_false', default=True)
    param("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    param('--use_huber_loss', action='store_true',
                        default=False, help="Whether to use Huber loss for critic update") # MSE/MAE loss function의 결합형태 
    param("--huber_delta", type=float, default=10.0) #MSE/MAE가 서로 전환되는 시점을 지정함. 

    # soft update parameters
    param('--use_soft_update', action='store_false',
                        default=True, help="Whether to use soft update")
    param('--tau', type=float, default=0.005,
                        help="Polyak update rate")
    # hard update parameters
    param('--hard_update_interval_episode', type=int, default=200,
                        help="After how many episodes the lagging target should be updated")
    param('--hard_update_interval', type=int, default=200,
                        help="After how many timesteps the lagging target should be updated")
    # rmatd3 parameters
    param("--target_action_noise_std", default=0.2, help="Target action smoothing noise for matd3")
    # rmasac parameters
    param('--alpha', type=float, default=1.0,
                        help="Initial temperature")
    param('--target_entropy_coef', type=float,
                        default=0.5, help="Initial temperature")
    param('--automatic_entropy_tune', action='store_false',
                        default=True, help="Whether use a centralized critic")
    # qmix parameters
    param('--use_double_q', action='store_false',
                        default=True, help="Whether to use double q learning")
    param('--hypernet_layers', type=int, default=2,
                        help="Number of layers for hypernetworks. Must be either 1 or 2")
    param('--mixer_hidden_dim', type=int, default=32,
                        help="Dimension of hidden layer of mixing network")
    param('--hypernet_hidden_dim', type=int, default=64,
                        help="Dimension of hidden layer of hypernetwork (only applicable if hypernet_layers == 2")
    param('--num_env_steps', type=int,
                        default=20000000, help="Number of env steps to train for")
    # exploration parameters
    param('--num_warmup_episodes', type=int, default=50,
                        help="Number of episodes to add to buffer with purely random actions")
    param('--epsilon_start', type=float, default=1.0,
                        help="Starting value for epsilon, for eps-greedy exploration")
    param('--epsilon_finish', type=float, default=0.05,
                        help="Ending value for epsilon, for eps-greedy exploration")
    param('--epsilon_anneal_time', type=int, default=50000,
                        help="Number of episodes until epsilon reaches epsilon_finish")
    param('--act_noise_std', type=float,
                        default=0.1, help="Action noise")

    # train parameters
    param('--actor_train_interval_step', type=int, default=2,
                        help="After how many critic updates actor should be updated")
    param('--train_interval_episode', type=int, default = 10,
                        help="Number of env steps between updates to actor/critic")
    param('--train_interval', type=int, default=10,
                        help="Number of episodes between updates to actor/critic")
    param("--use_value_active_masks",
                        action='store_true', default=False)

    # eval parameters
    param('--use_eval', action='store_false',
                        default=True, help="Whether to conduct the evaluation")
    param('--eval_interval', type=int,  default=1,
                        help="After how many episodes the policy should be evaled")
    param('--num_eval_episodes', type=int, default=1,
                        help="How many episodes to collect for each eval")

    # save parameters
    param('--save_interval', type=int, default=100000,
                        help="After how many episodes of training the policy model should be saved")

    # log parameters
    param('--log_interval', type=int, default=1000, #아직 구현되지 않음 
                        help="After how many episodes of training the policy model should be saved")

    # pretained parameters
    param("--model_dir", type=str, default=None)
    # VDN or QMIX
    param("--q_network_name", type=str, default=None)
    param("--mixer_name", type=str, default=None) # IF VDN: None

    param('--use_available_actions', action='store_true',
                        default=False, help="Whether to use available actions")
    param('--use_same_share_obs', action='store_false',
                        default=True, help="Whether to use available actions")
    param('--use_global_all_local_state', action='store_true',
                        default=False, help="Whether to use available actions")
    

    # save replay 
    param('--save_replay',
                        default=False, help="select to save a replay")
    param('--save_replay_interval', type=int,
                        default=100000, help="save term of replay")

    return parser
