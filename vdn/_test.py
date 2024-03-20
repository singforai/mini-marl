import torch

import torch.nn.functional as F

from _utils import cal_td_error

class Test:
    def __init__(self, test_env, args, device):

        self.test_env = test_env

        self.batch_size: int = args.batch_size
        self.chunk_size: int = args.chunk_size
        self.update_iter: int = args.update_iter
        self.test_episodes: int = args.test_episodes

        self.gamma: float = args.gamma
        self.grad_clip_norm: float = args.grad_clip_norm

        self.device = device

    def execute(self, behavior_network, target_network):
        score: float = 0
        iter_loss: float = 0.0
        for episode_i in range(self.test_episodes):
            state = self.test_env.reset()
            done = [False for _ in range(self.test_env.n_agents)]
            episode_loss: float = 0.0
            with torch.no_grad():
                hidden = behavior_network.init_hidden().to(self.device)
                target_hidden = target_network.init_hidden().to(self.device)
                while not all(done):
                    action, next_hidden, behavior_q = behavior_network.sample_action(torch.tensor(state).unsqueeze(0).to(self.device), hidden, epsilon=0)

                    next_state, reward, done, info = self.test_env.step(action[0])

                    target_q, next_target_hidden = target_network(torch.tensor(next_state).unsqueeze(dim=0).to(self.device), target_hidden)

                    td_error: float = cal_td_error(action=action[0], reward=reward, done=int(all(
                    done)), behavior_q=behavior_q, target_q=target_q, gamma=self.gamma)


                    episode_loss += td_error**2

                    score += sum(reward)
                    state = next_state
                    hidden = next_hidden
                    target_hidden = next_target_hidden

            iter_loss += episode_loss

        return score / self.test_episodes, iter_loss/self.test_episodes
