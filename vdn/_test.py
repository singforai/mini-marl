import torch

class Test:
    def __init__(self, test_env, args):

        self.test_env = test_env

        self.batch_size: int = args.batch_size
        self.chunk_size: int = args.chunk_size
        self.update_iter: int = args.update_iter
        self.test_episodes: int = args.test_episodes

        self.gamma: float = args.gamma
        self.grad_clip_norm: float = args.grad_clip_norm

    def execute(self, behavior_network):
        score: float = 0
        for episode_i in range(self.test_episodes):
            state = self.test_env.reset()
            done = [False for _ in range(self.test_env.n_agents)]
            with torch.no_grad():
                hidden = behavior_network.init_hidden()
                while not all(done):
                    action, next_hidden, _ = behavior_network.sample_action(
                        torch.tensor(state).unsqueeze(0), hidden, epsilon=0)
                    next_state, reward, done, info = self.test_env.step(action[0])
                    score += sum(reward)
                    state = next_state
                    hidden = next_hidden

        return score / self.test_episodes
