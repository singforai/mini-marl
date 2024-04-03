import numpy as np
import torch
import pdb
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.valuenorm import ValueNorm
from utils.util import check


class R_MAPPO:
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, actor_policy, critic_policy, device):

        self._use_popart: bool = args.use_popart
        self._use_valuenorm: bool = args.use_valuenorm
        self._use_huber_loss: bool = args.use_huber_loss
        self._use_max_grad_norm: bool = args.use_max_grad_norm
        self._use_recurrent_policy: bool = args.use_recurrent_policy
        self._use_clipped_value_loss: bool = args.use_clipped_value_loss
        self._use_value_active_masks: bool = args.use_value_active_masks
        self._use_naive_recurrent: bool = args.use_naive_recurrent_policy
        self._use_policy_active_masks: bool = args.use_policy_active_masks

        self.ppo_epoch: int = args.ppo_epoch
        self.training_batch_size: int = args.training_batch_size
        self.data_chunk_length: int = args.data_chunk_length

        self.clip_param: float = args.clip_param
        self.huber_delta: float = args.huber_delta
        self.entropy_coef: float = args.entropy_coef
        self.max_grad_norm: float = args.max_grad_norm
        self.value_loss_coef: float = args.value_loss_coef

        self.device: torch.device = device
        self.tpdv: dict = dict(dtype=torch.float32, device=device)

        self.actor_policy = actor_policy
        self.critic_policy = critic_policy 
        
        assert (
            self._use_popart and self._use_valuenorm
        ) == False, "self._use_popart and self._use_valuenorm can not be set True simultaneously"

        if self._use_popart:
            self.value_normalizer = self.critic_policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(input_shape = 1).to(self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(
        self, values, value_preds_batch, return_batch, active_masks_batch
    ):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )

        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = (
                self.value_normalizer.normalize(return_batch) - value_pred_clipped
            )
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (
                value_loss * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, actor_policy, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        (
            obs_batch,
            cent_obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.critic_policy.evaluate_actions(
            actor_policy = actor_policy,
            obs = obs_batch,
            cent_obs = cent_obs_batch, 
            rnn_states_actor = rnn_states_batch,
            rnn_states_critic = rnn_states_critic_batch,
            action = actions_batch,
            masks = masks_batch,
            available_actions = available_actions_batch,
            active_masks = active_masks_batch,
        )

        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss

        actor_policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                actor_policy.actor.parameters(), self.max_grad_norm
            )
            
        else:
            actor_grad_norm = get_gard_norm(actor_policy.actor.parameters())
        actor_policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch
        )

        self.critic_policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic_policy.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_gard_norm(self.critic_policy.critic.parameters())

        self.critic_policy.critic_optimizer.step()

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            imp_weights,
        )

    def train(self, central_buffer, central_obs, actor_policy, agent_id, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        buffer = central_buffer[agent_id]
        # central_advantage = []
        # for agent_buffer in central_buffer:
        #     if self._use_popart or self._use_valuenorm:

        #         advantages = agent_buffer.returns[:-1] - \
        #         self.value_normalizer.denormalize(agent_buffer.value_preds[:-1] )
        #         central_advantage.append(advantages)
        #     else:
        #         advantages = agent_buffer.returns[:-1] - agent_buffer.value_preds[:-1]
        #         central_advantage.append(advantages)
        # sum_advantage = np.sum(central_advantage, axis=0) #/ len(central_advantage)
        # advantages_copy = sum_advantage.copy()
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(
            buffer.value_preds[:-1]
            )
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        # mean_advantages =  np.mean(advantages_copy)
        # std_advantages = np.mean(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info["value_loss"] = 0
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["critic_grad_norm"] = 0
        train_info["ratio"] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    advantages = advantages, 
                    cent_obs = central_obs,
                    num_mini_batch = self.training_batch_size, 
                    data_chunk_length = self.data_chunk_length,
                    
                )
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(
                    advantages = advantages, 
                    cent_obs = central_obs,
                    num_mini_batch = self.training_batch_size,
                    
                )
            else:
                data_generator = buffer.feed_forward_generator(
                    advantages = advantages, 
                    cent_obs = central_obs,
                    num_mini_batch = self.training_batch_size,
                )

            for sample in data_generator:

                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                ) = self.ppo_update(sample = sample, actor_policy = actor_policy, update_actor = update_actor)
                
                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm.item()
                train_info["critic_grad_norm"] += critic_grad_norm.item()
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.training_batch_size

        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info

    def prep_training(self):
        self.actor_policy.actor.train()
        self.critic_policy.critic.train()

    def prep_rollout(self):
        self.actor_policy.actor.eval()
        self.critic_policy.critic.eval()