import wandb
import torch
import numpy as np

from gym import spaces

from replay_buffer.shared_buffer import SharedReplayBuffer
from runner.shared.observation_space import MultiAgentObservationSpace


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """

    def __init__(self, config):

        # config
        self.args = config["args"]
        self.train_env = config["train_env"]
        self.eval_env = config["eval_env"]
        self.device: torch.device = config["device"]
        self.num_agents: int = config["num_agents"]

        # name_parameter
        self.env_name: str = self.args.env_name
        self.algorithm_name: str = self.args.algorithm_name
        self.experiment_name: str = self.args.experiment_name

        # rollout_threads
        self.n_rollout_threads: int = self.args.n_rollout_threads
        self.n_eval_rollout_threads: int = self.args.n_eval_rollout_threads
        self.n_render_rollout_threads: int = self.args.n_render_rollout_threads

        # use_hyperparameter
        self.use_eval: bool = self.args.use_eval
        self.use_wandb: bool = self.args.use_wandb
        self.use_render: bool = self.args.use_render
        self.use_centralized_V: bool = self.args.use_centralized_V
        self.use_linear_lr_decay: bool = self.args.use_linear_lr_decay

        # parameters
        self.max_episodes: int = self.args.max_episodes
        self.episode_length: int = self.args.max_step
        self.hidden_size: int = self.args.hidden_size
        self.recurrent_N: int = self.args.recurrent_N
        self.batch_size: int = self.args.batch_size  # 현재는 1로 강제해야 함
        self.eval_episodes: int = self.args.eval_episodes

        # interval
        self.log_interval: int = self.args.log_interval
        self.save_interval: int = self.args.save_interval
        self.eval_interval: int = self.args.eval_interval

        # render time
        self.sleep_second: float = self.args.sleep_second

        # share_observation
        self.observation_space = self.train_env.observation_space
        if self.use_centralized_V:
            self._obs_high = np.tile(self.train_env._obs_high, self.num_agents)
            self._obs_low = np.tile(self.train_env._obs_low, self.num_agents)
            self.share_observation_space = MultiAgentObservationSpace(
                [
                    spaces.Box(self._obs_low, self._obs_high)
                    for _ in range(self.num_agents)
                ]
            )

        else:
            self.share_observation_space = self.observation_space

        from algorithms.ramppo_network import R_MAPPO as TrainAlgo
        from algorithms.policys.rmappo_policy import R_MAPPOPolicy as Policy

        # policy network
        self.policy = Policy(
            args=self.args,
            obs_space=self.observation_space[0],
            cent_obs_space=self.share_observation_space[0],
            act_space=self.train_env.action_space[0],
            device=self.device,
        )

        # algorithm
        self.trainer = TrainAlgo(args=self.args, policy=self.policy, device=self.device)

        # buffer
        self.buffer = SharedReplayBuffer(
            args=self.args,
            num_agents=self.num_agents,
            obs_space=self.observation_space[0],
            cent_obs_space=self.share_observation_space[0],
            act_space=self.train_env.action_space[0],
        )

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        """Train policies with data in buffer."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + "/actor.pt")
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + "/critic.pt")
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            if self.trainer._use_valuenorm:
                policy_vnorm_state_dict = torch.load(str(self.model_dir) + "/vnorm.pt")
                self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
