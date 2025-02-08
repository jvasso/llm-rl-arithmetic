from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast

# custom
from typing import Optional, Union
from tianshou.data import Batch

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torch.distributions.categorical import Categorical

from tianshou.data import ReplayBuffer, SequenceSummaryStats, to_torch_as
from tianshou.data.types import BatchWithAdvantagesProtocol, RolloutBatchProtocol
from tianshou.data.batch import Batch
from tianshou.policy import PGPolicy
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
from tianshou.policy.modelfree.pg import TDistFnDiscrOrCont
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.discrete import Actor as DiscreteActor
from tianshou.utils.net.discrete import Critic as DiscreteCritic


@dataclass(kw_only=True)
class A2CTrainingStats(TrainingStats):
    loss: SequenceSummaryStats
    actor_loss: SequenceSummaryStats
    vf_loss: SequenceSummaryStats
    ent_loss: SequenceSummaryStats
    # custom
    kl_div_loss: SequenceSummaryStats


TA2CTrainingStats = TypeVar("TA2CTrainingStats", bound=A2CTrainingStats)


# TODO: the type ignore here is needed b/c the hierarchy is actually broken! Should reconsider the inheritance structure.
class CustomA2CPolicy(PGPolicy[TA2CTrainingStats], Generic[TA2CTrainingStats]):  # type: ignore[type-var]
    """Implementation of Synchronous Advantage Actor-Critic. arXiv:1602.01783.

    :param actor: the actor network following the rules:
        If `self.action_type == "discrete"`: (`s_B` ->`action_values_BA`).
        If `self.action_type == "continuous"`: (`s_B` -> `dist_input_BD`).
    :param critic: the critic network. (s -> V(s))
    :param optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :param action_space: env's action space
    :param vf_coef: weight for value loss.
    :param ent_coef: weight for entropy loss.
    :param max_grad_norm: clipping gradients in back propagation.
    :param gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
    :param max_batchsize: the maximum size of the batch when computing GAE.
    :param discount_factor: in [0, 1].
    :param reward_normalization: normalize estimated values to have std close to 1.
    :param deterministic_eval: if True, use deterministic evaluation.
    :param observation_space: the space of the observation.
    :param action_scaling: if True, scale the action from [-1, 1] to the range of
        action_space. Only used if the action_space is continuous.
    :param action_bound_method: method to bound action to range [-1, 1].
        Only used if the action_space is continuous.
    :param lr_scheduler: if not None, will be called in `policy.update()`.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        *,
        actor: torch.nn.Module | ActorProb | DiscreteActor,
        critic: torch.nn.Module | Critic | DiscreteCritic,
        optim: torch.optim.Optimizer,
        dist_fn: TDistFnDiscrOrCont,
        action_space: gym.Space,
        kl_approx_method:str,
        actor_sft: torch.nn.Module, # custom
        kl_div_coef: float = 0.1, # custom
        certainty_estimation:str=None,
        certainty_scaling_exponent: float = None,
        certainty_scaling_intercept: float = None,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float | None = None,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        discount_factor: float = 0.99,
        # TODO: rename to return_normalization?
        reward_normalization: bool = False,
        deterministic_eval: bool = False,
        observation_space: gym.Space | None = None,
        action_scaling: bool = True,
        action_bound_method: Literal["clip", "tanh"] | None = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        super().__init__(
            actor=actor,
            optim=optim,
            dist_fn=dist_fn,
            action_space=action_space,
            discount_factor=discount_factor,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        self.critic = critic
        assert 0.0 <= gae_lambda <= 1.0, f"GAE lambda should be in [0, 1] but got: {gae_lambda}"
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.max_grad_norm = max_grad_norm
        self.max_batchsize = max_batchsize
        self._actor_critic = ActorCritic(self.actor, self.critic)

        # custom
        self.kl_approx_method = kl_approx_method
        self.actor_sft        = actor_sft
        self.kl_div_coef      = kl_div_coef

        # certainty
        self.certainty_estimation        = certainty_estimation
        self.certainty_scaling_exponent  = certainty_scaling_exponent
        self.certainty_scaling_intercept = certainty_scaling_intercept

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        return batch

    def _compute_returns(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                v_s.append(self.critic(minibatch.obs))
                v_s_.append(self.critic(minibatch.obs_next))
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Empirical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        # TODO: see todo in PGPolicy.process_fn
        if self.rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        if self.rew_norm:
            batch.returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return cast(BatchWithAdvantagesProtocol, batch)

    # TODO: mypy complains b/c signature is different from superclass, although
    #  it's compatible. Can this be fixed?
    def learn(  # type: ignore
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
        *args: Any,
        **kwargs: Any,
    ) -> TA2CTrainingStats:
        losses, actor_losses, vf_losses, ent_losses, kl_div_list = [], [], [], [], [] # custom
        split_batch_size = batch_size or -1
        for _ in range(repeat):
            for minibatch in batch.split(split_batch_size, merge_last=True):
                # calculate loss for actor
                dist = self(minibatch).dist
                log_prob_actions = dist.log_prob(minibatch.act)
                log_prob_actions = log_prob_actions.reshape(len(minibatch.adv), -1).transpose(0, 1)
                actor_loss = -(log_prob_actions * minibatch.adv).mean()
                # calculate loss for critic
                value = self.critic(minibatch.obs).flatten()
                vf_loss = F.mse_loss(minibatch.returns, value)
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()

                # custom
                if self.kl_div_coef != 0:
                    kl_div = self.compute_kl_div_penalty(minibatch=minibatch, dist=dist, log_probs_actions=log_prob_actions)
                    loss = actor_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss + self.kl_div_coef*kl_div
                else:
                    kl_div = None
                    loss = actor_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                self.optim.zero_grad()
                loss.backward()
                if self.max_grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(),
                        max_norm=self.max_grad_norm,
                    )
                self.optim.step()
                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                if kl_div is not None: kl_div_list.append(kl_div.item())  # custom
                losses.append(loss.item())

                # # debug
                # if kl_div is not None:
                #     if len(kl_div_list)<=20:
                #         print(kl_div.item())
                #     else:
                #         print(np.mean(kl_div_list[-20:]))

        loss_summary_stat = SequenceSummaryStats.from_sequence(losses)
        actor_loss_summary_stat = SequenceSummaryStats.from_sequence(actor_losses)
        vf_loss_summary_stat = SequenceSummaryStats.from_sequence(vf_losses)
        ent_loss_summary_stat = SequenceSummaryStats.from_sequence(ent_losses)

        kl_div_loss_stat = SequenceSummaryStats.from_sequence(kl_div_list) # custom

        return A2CTrainingStats(  # type: ignore[return-value]
            loss=loss_summary_stat,
            actor_loss=actor_loss_summary_stat,
            vf_loss=vf_loss_summary_stat,
            ent_loss=ent_loss_summary_stat,
            kl_div_loss=kl_div_loss_stat # custom
        )
    

    def compute_kl_div_penalty(self, minibatch:Batch, dist:Categorical, log_probs_actions:torch.Tensor):
        probs_actions, probs_actions_sft = None, None
        
        # get sft predictions
        with torch.no_grad():
            output_sft = self.forward_sft(minibatch)
            dist_sft            :Categorical = output_sft.dist
            dist_sft_weighted_kl:Categorical = output_sft.dist_weighted_kl if self.certainty_estimation is not None else None
        
        log_probs_actions_sft = dist_sft.log_prob(minibatch.act)
        log_probs_actions_sft = log_probs_actions_sft.reshape(len(minibatch.adv), -1).transpose(0, 1)

        # estimate certainty
        batch_indices   = torch.arange(dist_sft.probs.size(0))
        certainty_coeff = self.compute_certainty_coeff(dist_sft_weighted_kl=dist_sft_weighted_kl, batch_indices=batch_indices, minibatch=minibatch)

        # compute KL
        if self.kl_approx_method == 'k1':
            kl_div_penalty = torch.mean(certainty_coeff * (log_probs_actions-log_probs_actions_sft))
        elif self.kl_approx_method == 'k2': # KL[p,p_old] : 1/2*(log p_old(x) - log p(x))^2 and x ~ p
            kl_div_penalty = torch.mean(certainty_coeff * 0.5 * (log_probs_actions_sft - log_probs_actions)**2 )
        elif self.kl_approx_method == 'k3': # KL[p,p_old] : (r−1)−log(r) with r = p_old(x)/p(x) and x ~ p
            probs_actions     = self.extract_probs_actions(dist=dist    , minibatch=minibatch) if probs_actions     is None else probs_actions
            probs_actions_sft = self.extract_probs_actions(dist=dist_sft, minibatch=minibatch) if probs_actions_sft is None else probs_actions_sft
            r = probs_actions_sft/probs_actions
            log_r = log_probs_actions_sft - log_probs_actions
            kl_div_penalty = torch.mean(certainty_coeff * (r-1 - log_r))
        else:
            raise NotImplementedError()
        
        return kl_div_penalty
    

    def extract_probs_actions(self, dist:Categorical, minibatch:Batch):
        return dist.probs.gather(1, minibatch.act.long().unsqueeze(1)).transpose(0,1)
    

    def compute_certainty_coeff(self, dist_sft_weighted_kl, batch_indices, minibatch):
        if self.certainty_estimation is None:
            return 1
        else:
            certainty = self.estimate_certainty(dist_sft_weighted_kl=dist_sft_weighted_kl, batch_indices=batch_indices, minibatch=minibatch)
            scaled_certainty = self.scale_certainty(certainty)
        return scaled_certainty
    

    def scale_certainty(self, certainty:torch.Tensor):
        assert self.certainty_scaling_intercept <= 0
        slope = 1 - self.certainty_scaling_intercept # make sure certainty(x=1)=1
        return slope * (certainty**self.certainty_scaling_exponent) + self.certainty_scaling_intercept
    

    def estimate_certainty(self, dist_sft_weighted_kl, batch_indices, minibatch):
        probs_sft = dist_sft_weighted_kl.probs[batch_indices, :]

        if self.certainty_estimation=='max_prob':
            max_probs, _ = torch.max(probs_sft, dim=1)
            certainty = max_probs
        elif self.certainty_estimation=='negentropy_max_prob':
            max_probs, _ = torch.max(probs_sft, dim=1)
            support_size = torch.tensor(probs_sft.shape[-1])
            certainty = 1 + torch.log(max_probs)/torch.log(support_size)
        elif self.certainty_estimation=='negentropy_sample':
            certainty = 1 + torch.log()
            raise NotImplementedError()
        elif self.certainty_estimation=='negentropy':
            # log probs etc.
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        # # debug
        # critical_tokens_idx = []
        # for env_idx in range(len(minibatch['info'])):
        #     log_dict_path = minibatch['info']['log_dict_path'][env_idx]
        #     if log_dict_path is not None:
        #         critical_tokens_idx.append(env_idx)
        # print(f'certainty: {certainty}  ||  {critical_tokens_idx}')
        
        return certainty



    def forward_sft(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        probas_dict, hidden = self.actor_sft(batch.obs, state=state, info=batch.info)
        logits = probas_dict['probas']
        logits_weighted_kl = probas_dict['probas_weighted_kl']
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)

        if self.certainty_estimation is not None:
            if isinstance(logits_weighted_kl, tuple):
                dist_weighted_kl = self.dist_fn(*logits_weighted_kl)
            else:
                dist_weighted_kl = self.dist_fn(logits_weighted_kl)
            
            return Batch(logits=logits, dist=dist, dist_weighted_kl=dist_weighted_kl)
        else:
            return Batch(logits=logits, dist=dist)
    

        # if self.deterministic_eval and not self.training:
        #     if self.action_type == "discrete":
        #         act = logits.argmax(-1)
        #     elif self.action_type == "continuous":
        #         act = logits[0]
        # else:
        #     act = dist.sample()
        # return Batch(logits=logits, act=act, state=hidden, dist=dist)


    # def compute_kl_div_loss(self, minibatch, log_prob):
    #     lob_prob_sft, entropy_sft = self.get_lob_prob_sft(minibatch=minibatch)
    #     kl_div_loss = self.compute_kl_div_penalty(log_prob_rl=log_prob, log_prob_sft=lob_prob_sft, entropy_sft=entropy_sft)
    #     return kl_div_loss
    

    # def compute_kl_div_penalty(self, log_prob_rl, log_prob_sft, entropy_sft):
    #     # formula: E_{(x, y) \sim D_{\pi_\phi^{\mathrm{RL}}}} \log \left(\pi_\phi^{\mathrm{RL}}(y \mid x) / \pi^{\mathrm{SFT}}(y \mid x)\right)
    #     if self.old_policy_trick is None:
    #         mean_log_ratio = torch.mean(log_prob_rl-log_prob_sft)
    #     elif self.old_policy_trick==1:
    #         raise NotImplementedError()
    #         torch.mean((1+entropy_sft)**(-self.old_policy_coef)*(log_prob_rl-log_prob_sft))
    #     elif self.old_policy_trick==2:
    #         mean_log_ratio = torch.mean(log_prob_sft(log_prob_rl-log_prob_sft))
    #     return mean_log_ratio
    

    # def get_lob_prob_sft(self, minibatch):
    #     with torch.no_grad():
    #         dist_sft:torch.distributions.categorical.Categorical = self.forward_sft(minibatch).dist
    #     log_prob_sft = dist_sft.log_prob(minibatch.act)
    #     log_prob_sft = log_prob_sft.reshape(len(minibatch.adv), -1).transpose(0, 1)

    #     if self.old_policy_trick==1:
    #         entropy_sft = dist_sft.entropy()
    #         prob_sft    = None
    #     elif self.old_policy_trick==2:
    #         entropy_sft = None
    #         prob_sft    = dist_sft.probs
    #     else:
    #         entropy_sft = None
    #     return log_prob_sft, entropy_sft, prob_sft