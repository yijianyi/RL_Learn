from __future__ import annotations

################################
# answer code
import math
################################
from typing import Dict

import torch

################################
# answer code
from llm_rl_final_proj.models.logprobs import (
    approx_kl_from_logprobs, compute_per_token_logprobs,
    masked_mean, masked_mean_per_row,
)
################################
from llm_rl_final_proj.rl.base import RLAlgorithm
################################
# answer code
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch, iter_minibatches
from llm_rl_final_proj.utils.torch_utils import clip_grad_norm_
################################


class GRPO(RLAlgorithm):
    ################################
    # answer code
    """Token-level PPO clipping with a scalar advantage per completion."""
    ################################

    name = "grpo"

    ################################
    # answer code
    def policy_objective(self, log_ratio, advantages, mask):
        high = self.cfg.clip_eps_high if self.cfg.clip_eps_high > 0 else self.cfg.clip_eps
        ratio = log_ratio.clamp(-20, 20).exp()
        clipped = ratio.clamp(1 - self.cfg.clip_eps, 1 + high)
        objective = torch.minimum(ratio * advantages[:, None], clipped * advantages[:, None])
        pg_loss = -masked_mean_per_row(objective, mask).mean()
        clipfrac = masked_mean(((ratio < 1 - self.cfg.clip_eps) | (ratio > 1 + high)).float(), mask)
        return pg_loss, clipfrac
    ################################

    ################################
    # answer code
    def update(self, model, optimizer, rollout, grad_accum_steps=1) -> Dict[str, float]:
        cfg = self.cfg
        model.train()
        model.config.use_cache = False
        accumulation = max(1, grad_accum_steps)
        params = [p for p in model.parameters() if p.requires_grad]
        totals = dict(loss=0.0, kl=0.0, entropy=0.0, clipfrac=0.0)
        n_mb = opt_steps = pending = skipped_empty = skipped_nonfinite = 0
        total_grad_norm = 0.0
        rng = torch.Generator(device=rollout.input_ids.device)
        rng.manual_seed(self._next_update_seed())
        optimizer.zero_grad(set_to_none=True)

        def step_optimizer(count):
            # Correct the last, partially filled accumulation window.
            if count < accumulation:
                for p in params:
                    if p.grad is not None:
                        p.grad.mul_(accumulation / count)
            norm = clip_grad_norm_(params, cfg.max_grad_norm)
            valid = math.isfinite(norm)
            if valid:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            return valid, norm

        for _ in range(cfg.ppo_epochs):
            for mb in iter_minibatches(rollout, cfg.minibatch_size, shuffle=True,
                                      generator=rng, device=next(model.parameters()).device):
                mask = mb.completion_mask
                if mask.sum().item() == 0:
                    skipped_empty += 1
                    continue
                adv = mb.advantages.detach().clamp(-cfg.adv_clip, cfg.adv_clip)
                new_logp = compute_per_token_logprobs(model, mb.input_ids, mb.attention_mask)
                pg_loss, clipfrac = self.policy_objective(new_logp - mb.old_logprobs.detach(), adv, mask)
                kl = approx_kl_from_logprobs(new_logp, mb.ref_logprobs.detach(), mask)
                entropy = -masked_mean(new_logp, mask)
                loss = pg_loss + cfg.kl_coef * kl
                if not torch.isfinite(loss):
                    skipped_nonfinite += 1
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
                    continue
                (loss / accumulation).backward()
                pending += 1
                if pending == accumulation:
                    valid, norm = step_optimizer(pending)
                    pending = 0
                    if valid:
                        opt_steps += 1
                        total_grad_norm += norm
                    else:
                        skipped_nonfinite += 1
                for key, value in (("loss", loss), ("kl", kl), ("entropy", entropy), ("clipfrac", clipfrac)):
                    totals[key] += value.detach().item()
                n_mb += 1
        if pending:
            valid, norm = step_optimizer(pending)
            if valid:
                opt_steps += 1
                total_grad_norm += norm
            else:
                skipped_nonfinite += 1
        denom = max(1, n_mb)
        return {
            "train/policy_loss_with_kl_penalty_mean_over_minibatches": totals["loss"] / denom,
            "train/approximate_kl_divergence_policy_vs_reference_mean_over_minibatches": totals["kl"] / denom,
            "train/policy_token_entropy_mean_over_minibatches": totals["entropy"] / denom,
            "train/fraction_of_completion_tokens_where_ppo_ratio_was_clipped_mean_over_minibatches": totals["clipfrac"] / denom,
            "train/count_minibatches_skipped_because_completion_mask_had_no_tokens": float(skipped_empty),
            "train/count_update_attempts_skipped_due_to_nonfinite_loss_or_gradients": float(skipped_nonfinite),
            "train/gradient_global_norm_after_clipping_mean_over_optimizer_steps": total_grad_norm / max(1, opt_steps),
            "train/count_optimizer_steps_per_training_iteration": float(opt_steps),
        }
    ################################
