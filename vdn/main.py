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

from _test import Test
from replay_buffer.per import Prioritized_Experience_Replay

def main(args):
    parser = get_config()
    args = parser.parse_known_args(args)[0]

    use_wandb: bool = args.use_wandb
    use_render: bool = args.use_render
    use_time_sleep: bool = args.use_time_sleep

    n_buffer: int = 0
    count_step: int = 0
    chunk_size: int = args.chunk_size
    max_episodes: int = args.max_episodes
    warm_up_chunk: int = args.buffer_limit
    test_interval: int = args.test_interval
    update_target_interval: int = args.update_target_interval

    score: float = 0
    gamma: float = args.gamma
    max_epsilon: float = args.max_epsilon
    min_epsilon: float = args.min_epsilon
    sleep_second: float = args.sleep_second if use_render else None

    environment: str = args.env_name
    experiment_name: str = args.experiment_name

    if use_wandb:
        import wandb
        wandb.init(entity=args.entity_name, 
                   project= environment.split(":")[1],
                   name=f"{experiment_name}-{int(time.time())}",
                   config = args, reinit=True)
 
    if args.use_cuda and torch.cuda.is_available():
        torch.set_num_threads(args.n_training_threads)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    set_logging(experiment_name = experiment_name)
    log_hyperparameter(args = args, device = device)

    fix_random_seed(args.seed) if args.fix_seed else None

    train_env = gym.make(
        id=environment, max_steps=args.max_step, step_cost=args.step_cost)
    test_env = gym.make(
        id=environment, max_steps=args.max_step, step_cost=args.step_cost)

    Replay_buffer = Prioritized_Experience_Replay(args=args)

    target_network, behavior_network = action_network(args, train_env, device)
    target_network.load_state_dict(state_dict=behavior_network.state_dict())
    
    optimizer = optim.Adam(params=behavior_network.parameters(), lr=args.lr)

    target_module = target_setting(args = args, device = device)

    test = Test(test_env, args, device)

    state_chunk, action_chunk, reward_chunk, next_state_chunk, done_chunk, td_error_chunk = chunk_initialize()

    pbar = tqdm(total=warm_up_chunk, desc="warm up", ncols=70)
    while warm_up_chunk > n_buffer:
        epsilon: float = 1
        state = train_env.reset()
        done = [False for _ in range(train_env.n_agents)]

        with torch.no_grad():
            hidden = behavior_network.init_hidden().to(device)
            target_hidden = target_network.init_hidden().to(device)
            while not all(done):
                action, next_hidden, behavior_q = behavior_network.sample_action(obs=torch.Tensor(state).unsqueeze(dim=0).to(device), hidden=hidden, epsilon=epsilon)
                next_state, reward, done, info = train_env.step(action[0])

                target_q, next_target_hidden  = target_network(torch.tensor(next_state).unsqueeze(dim=0).to(device), target_hidden)

                td_error: float = cal_td_error(action=action[0], reward=reward, done=int(all(
                    done)), behavior_q=behavior_q, target_q=target_q, gamma=gamma)

                count_step += 1

                state_chunk.append(state)
                action_chunk.append(action.tolist())
                reward_chunk.append(reward)
                next_state_chunk.append(next_state)
                done_chunk.append(int(all(done)))
                td_error_chunk.append(td_error)

                if count_step % chunk_size == 0:
                    sample = [state_chunk, action_chunk, reward_chunk, next_state_chunk, done_chunk]
                    td_error = sum(td_error_chunk)
                    n_buffer = Replay_buffer.collect_sample(sample=sample, td_error=td_error, warm_up = True)
                    state_chunk, action_chunk, reward_chunk, next_state_chunk, done_chunk, td_error_chunk = chunk_initialize()
                    pbar.update(1)
                    
                    if n_buffer == warm_up_chunk:
                        count_step = 0
                        break
                    
                state = next_state
                hidden = next_hidden
                target_hidden = next_target_hidden
    
    

    for episode in tqdm(range(max_episodes), desc="training", ncols=70):
        #pdb.set_trace()
        train_env.render() if use_render else None

        state = train_env.reset()
        done = [False for _ in range(train_env.n_agents)]
        epsilon: float = max(min_epsilon, max_epsilon - (max_epsilon -min_epsilon) * (episode / args.epsilon_anneal_episode))

        with torch.no_grad():
            hidden = behavior_network.init_hidden().to(device)
            target_hidden = target_network.init_hidden().to(device)
            while not all(done):
                action, next_hidden, behavior_q = behavior_network.sample_action(obs=torch.tensor(state).unsqueeze(dim=0).to(device), hidden=hidden, epsilon=epsilon)

                next_state, reward, done, info = train_env.step(action[0])

                target_q, next_target_hidden  = target_network(torch.tensor(next_state).unsqueeze(dim=0).to(device), target_hidden)

                td_error = cal_td_error(action=action[0], reward=reward, done=int(all(done)), behavior_q=behavior_q, target_q=target_q, gamma=gamma)
                
                count_step += 1

                state_chunk.append(state)
                action_chunk.append(action.tolist())
                reward_chunk.append(reward)
                next_state_chunk.append(next_state)
                done_chunk.append(int(all(done)))
                td_error_chunk.append(td_error)

                if count_step % chunk_size == 0:
                    sample = [state_chunk, action_chunk, reward_chunk, next_state_chunk, done_chunk]
                    td_error = sum(td_error_chunk)
                    Replay_buffer.collect_sample(sample=sample, td_error=td_error, warm_up = False)

                    state_chunk, action_chunk, reward_chunk, next_state_chunk, done_chunk, td_error_chunk = chunk_initialize()
                
                state = next_state
                hidden = next_hidden
                target_hidden = next_target_hidden

                score += sum(reward)
                if use_render:
                    train_env.render()
                    if use_time_sleep:
                        time.sleep(sleep_second)

        target_module.train(Replay_buffer=Replay_buffer, behavior_network=behavior_network,
                        target_network=target_network, optimizer=optimizer, epsilon=epsilon)

        if episode % update_target_interval == 0:
            target_network.load_state_dict(state_dict=behavior_network.state_dict())

        if (episode + 1) % test_interval == 0:
            test_score: float = test.execute(behavior_network=behavior_network)
            train_score: float = score / test_interval
            
            logging.info(f"{(episode+1):<8}/{max_episodes:<10} episodes | avg train score: {train_score:<6.2f} | avg test score: {test_score:<6.2f} | n_buffer: {n_buffer:<6} | eps: {epsilon:<5.4f} | alpha: {Replay_buffer.alpha:<6.4f} | beta: {Replay_buffer.beta:<6.4f}")
            if use_wandb: wandb.log({'episode': episode, 'train-score': train_score, 'test-score': test_score,'epsilon': epsilon, "alpha": Replay_buffer.alpha, "beta": Replay_buffer.beta}) 
            
            score: float = 0.0

    train_env.close()
    test_env.close()


if __name__ == '__main__':
    main(sys.argv[1:])
