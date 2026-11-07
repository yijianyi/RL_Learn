################################
# answer code
from __future__ import annotations

from dataclasses import replace
import torch
from llm_rl_final_proj.rl.grpo import GRPO


class PPO(GRPO):
    """PPO with a learned prompt-value baseline for terminal sequence rewards.

    Treat a completion as a contextual-bandit episode. A small critic predicts its
    reward from frozen, mean-pooled prompt embeddings; the actor uses token PPO
    clipping. This is a simple Part 2 extension, without a token-value/GAE model.
    """

    name = "ppo"

    def update(self, model, optimizer, rollout, grad_accum_steps=1):
        rollout = rollout.to(next(model.parameters()).device)
        with torch.no_grad():
            prompt_mask = rollout.attention_mask.float().clone()
            prompt_mask[:, 1:] *= 1 - rollout.completion_mask
            embeddings = model.get_input_embeddings()(rollout.input_ids).float()
            features = ((embeddings * prompt_mask[:, :, None]).sum(dim=1)
                        / prompt_mask.sum(dim=1, keepdim=True).clamp_min(1))
        if not hasattr(self, "value_head"):
            self.value_head = torch.nn.Sequential(
                torch.nn.Linear(features.shape[-1], 128), torch.nn.Tanh(),
                torch.nn.Linear(128, 1),
            ).to(features.device)
            self.value_optimizer = torch.optim.Adam(self.value_head.parameters(),
                                                     lr=optimizer.param_groups[0]["lr"])
        for group in self.value_optimizer.param_groups:
            group["lr"] = optimizer.param_groups[0]["lr"]
        with torch.no_grad():
            old_values = self.value_head(features).squeeze(-1)
            advantages = rollout.rewards.detach() - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
        value_losses = []
        for _ in range(self.cfg.ppo_epochs):
            for start in range(0, features.shape[0], self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                values = self.value_head(features[start:end]).squeeze(-1)
                loss = (values - rollout.rewards[start:end].detach()).square().mean()
                self.value_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.cfg.max_grad_norm)
                if torch.isfinite(norm):
                    self.value_optimizer.step()
                value_losses.append(loss.detach().item())
        metrics = super().update(model, optimizer, replace(rollout, advantages=advantages), grad_accum_steps)
        metrics["train/value_loss"] = sum(value_losses) / max(1, len(value_losses))
        metrics["train/value_mean"] = old_values.mean().item()
        return metrics
################################
