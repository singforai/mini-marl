import numpy as np
import torch
import pdb
import torch.nn as nn
from utils.util import get_grad_norm, huber_loss, mse_loss, split_chunks_into_n_ags
from utils.valuenorm import ValueNorm
from utils.util import check
from utils.layers.qmix import QMixNet


class R_MAPPO:
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, policy, device):

        self.args = args

        self.use_adv_noise = False
        self.alpha = 0.1

        self.n_agents = 2

        self.device: torch.device = device
        self.tpdv: dict = dict(dtype=torch.float32, device=device)

        self.policy = policy

        args.state_shape = policy.share_obs_space.shape[0]  # obs_space is (98,)

        assert (
            self.args.use_popart and self.args.use_valuenorm
        ) == False, "self._use_popart and self._use_valuenorm can not be set True simultaneously"

        if self.args.use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self.args.use_valuenorm:
            self.value_normalizer = ValueNorm(input_shape=1).to(self.device)
        else:
            self.value_normalizer = None

        if args.use_coordinate and args.if_AMix:
            self.amix_net = QMixNet(args)
            if args.use_cuda:
                self.amix_net = self.amix_net.cuda()

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.args.clip_param, self.args.clip_param
        )

        if self.args.use_popart or self.args.use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self.args.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.args.huber_delta)
            value_loss_original = huber_loss(error_original, self.args.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.args.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self.args.use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True, noise_vector=None):
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
            share_obs_batch,
            obs_batch,
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

        share_obs_batch = torch.from_numpy(share_obs_batch)

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Add noise to advantage values

        # if self.use_adv_noise:
        #    adv_noise = check(noise_vector).to(**self.tpdv)
        #    adv_noise = adv_noise[:, 0:1].repeat(adv_targ.shape[0] // 2, 1).view(-1, 1)  # 2 is n_agents
        #    adv_targ = (1 - self.alpha) * adv_targ + self.alpha * adv_noise

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        if self.args.use_coordinate:
            prcl_clip_param = self.args.clip_param
            adv_targ_ags = split_chunks_into_n_ags(self.args, adv_targ)
            imp_weights_ags = split_chunks_into_n_ags(self.args, imp_weights)
            active_masks_batch_ags = split_chunks_into_n_ags(self.args, active_masks_batch)
            state_batch_ags = split_chunks_into_n_ags(self.args, share_obs_batch)
            if self.args.use_cuda:
                state_batch_ags = state_batch_ags.cuda()
                # for surr1
            ratio = torch.prod(imp_weights_ags, dim=-2, keepdim=True).repeat(1, 1, self.n_agents, 1)
            # for surr2
            if self.args.if_double_clip:
                prod_ratios_for_surr2 = []
                for i in range(self.n_agents):
                    ratios = imp_weights_ags.clone()
                    ratios[:, :, i, :] = torch.ones_like(ratios[:, :, i, :])
                    prod_others_ratio_i = torch.prod(ratios, dim=-2).squeeze()
                    inner_eps = self.args.double_clip_inner_eps
                    prod_others_ratio_i_for_surr2 = torch.clamp(prod_others_ratio_i, 1 - inner_eps, 1 + inner_eps)
                    prod_others_i_for_surr2 = prod_others_ratio_i_for_surr2 * imp_weights_ags[:, :, i, :].squeeze()
                    prod_ratios_for_surr2.append(prod_others_i_for_surr2)
                ratio_for_surr2 = torch.stack(prod_ratios_for_surr2, dim=-1).unsqueeze(-1)
            if self.args.clip_before_prod:
                clipped_ratio = torch.prod(
                    torch.clamp(imp_weights_ags, 1.0 - self.args.clpr_clip_param, 1.0 + self.args.clpr_clip_param),
                    dim=-2,
                    keepdim=True,
                ).repeat(1, 1, self.n_agents, 1)
            else:
                if self.args.if_double_clip:
                    clipped_ratio = torch.clamp(ratio_for_surr2, 1.0 - prcl_clip_param, 1.0 + prcl_clip_param)
                else:
                    clipped_ratio = torch.clamp(ratio, 1.0 - prcl_clip_param, 1.0 + prcl_clip_param)

            surr1 = ratio * adv_targ_ags
            surr2 = clipped_ratio * adv_targ_ags

            if self.args.if_AMix:
                if self.args.min_before_mix:
                    surr = self.amix_net(
                        torch.min(surr1, surr2) * active_masks_batch_ags / active_masks_batch.sum(), state_batch_ags
                    )
                else:
                    surr1 = self.amix_net(surr1 * active_masks_batch_ags / active_masks_batch.sum(), state_batch_ags)
                    surr2 = self.amix_net(surr2 * active_masks_batch_ags / active_masks_batch.sum(), state_batch_ags)
                    surr = torch.min(surr1, surr2)
            else:  # ASum
                surr = torch.min(surr1, surr2) * active_masks_batch_ags / active_masks_batch.sum()
                surr = surr.sum(-2)
            policy_loss = -surr.sum()
            self.policy.actor_optimizer.zero_grad()
            if update_actor:
                (policy_loss - dist_entropy * self.args.entropy_coef).backward()
            if self.args.use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.args.max_grad_norm)
            else:
                actor_grad_norm = get_grad_norm(self.policy.actor.parameters())
            self.policy.actor_optimizer.step()
        else:  # MAPPO
            surr1 = imp_weights * adv_targ
            surr2 = torch.clamp(imp_weights, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * adv_targ

            if self.args.use_policy_active_masks:
                policy_action_loss = (
                    -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
                ).sum() / active_masks_batch.sum()
            else:
                policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

            policy_loss = policy_action_loss

            self.policy.actor_optimizer.zero_grad()

            if update_actor:
                (policy_loss - dist_entropy * self.args.entropy_coef).backward()

            if self.args.use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.args.max_grad_norm)
            else:
                actor_grad_norm = get_grad_norm(self.policy.actor.parameters())

            self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.args.value_loss_coef).backward()

        if self.args.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.args.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            imp_weights,
        )

    def train(self, buffer, update_actor=True, noise_vector=None):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        if self.args.use_popart or self.args.use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
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

        for _ in range(self.args.ppo_epoch):
            if self.args.use_recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    advantages=advantages,
                    num_mini_batch=self.args.training_batch_size,
                    data_chunk_length=self.args.data_chunk_length,
                )
            elif self.args.use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(
                    advantages=advantages, num_mini_batch=self.args.training_batch_size
                )
            else:
                data_generator = buffer.feed_forward_generator(
                    advantages=advantages, num_mini_batch=self.args.training_batch_size
                )

            for sample in data_generator:

                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                ) = self.ppo_update(sample=sample, update_actor=update_actor, noise_vector=noise_vector)

                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm.item()
                train_info["critic_grad_norm"] += critic_grad_norm.item()
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.args.ppo_epoch * self.args.training_batch_size
        train_info["Sampling_Rewards"] = buffer.rewards.sum()

        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
