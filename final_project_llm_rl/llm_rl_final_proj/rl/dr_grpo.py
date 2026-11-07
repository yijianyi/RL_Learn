from __future__ import annotations

################################
# answer code
from llm_rl_final_proj.rl.grpo import GRPO
from llm_rl_final_proj.models.logprobs import masked_mean, masked_mean_per_row
################################
import torch


################################
# answer code
class DrGRPO(GRPO):
    name = "dr_grpo"

    def policy_objective(self, log_ratio, advantages, mask):
        ratio = log_ratio.clamp(-20, 20).exp()
        high = self.cfg.clip_eps_high if self.cfg.clip_eps_high > 0 else self.cfg.clip_eps
        clipped = ratio.clamp(1 - self.cfg.clip_eps, 1 + high)
        objective = torch.minimum(ratio * advantages[:, None], clipped * advantages[:, None])
        # Every token has the same weight, independent of its completion's length.
        loss = -(objective * mask).sum(dim=1).mean() / max(1, self.cfg.max_completion_length)
        clipfrac = masked_mean(((ratio < 1 - self.cfg.clip_eps) | (ratio > 1 + high)).float(), mask)
        return loss, clipfrac
################################
