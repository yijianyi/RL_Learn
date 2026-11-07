from __future__ import annotations

################################
# answer code
from llm_rl_final_proj.rl.grpo import GRPO
from llm_rl_final_proj.models.logprobs import masked_mean, masked_mean_per_row
################################
import torch


################################
# answer code
class GSPO(GRPO):
    name = "gspo"

    def policy_objective(self, log_ratio, advantages, mask):
        log_ratio = masked_mean_per_row(log_ratio, mask).clamp(-20, 20)
        ratio = log_ratio.exp()
        high = self.cfg.clip_eps_high if self.cfg.clip_eps_high > 0 else self.cfg.clip_eps
        clipped = ratio.clamp(1 - self.cfg.clip_eps, 1 + high)
        loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
        valid = (mask.sum(dim=1) > 0).float()
        clipfrac = ((((ratio < 1 - self.cfg.clip_eps) | (ratio > 1 + high)).float() * valid).sum()
                    / valid.sum().clamp_min(1))
        return loss, clipfrac
################################
