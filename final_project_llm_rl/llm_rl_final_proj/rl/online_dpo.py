################################
# answer code
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from llm_rl_final_proj.models.logprobs import compute_per_token_logprobs
from llm_rl_final_proj.rl.base import RLAlgorithm
from llm_rl_final_proj.utils.torch_utils import clip_grad_norm_


class OnlineDPO(RLAlgorithm):
    """Create best/worst pairs within each on-policy prompt group, then apply DPO."""

    name = "online_dpo"

    def __init__(self, cfg, *, group_size=4, beta=0.1):
        super().__init__(cfg)
        if group_size < 2 or beta <= 0:
            raise ValueError("Online DPO requires group_size >= 2 and beta > 0")
        self.group_size, self.beta = group_size, beta

    def update(self, model, optimizer, rollout, grad_accum_steps=1):
        n = rollout.rewards.numel()
        if n % self.group_size:
            raise ValueError("Reward count must be divisible by group_size")
        pairs = []
        for start in range(0, n, self.group_size):
            scores = rollout.rewards[start:start + self.group_size]
            best, worst = scores.argmax().item(), scores.argmin().item()
            if scores[best] > scores[worst]:
                pairs.append((start + best, start + worst))
        if not pairs:
            return {"train/online_dpo_loss": 0.0, "train/online_dpo_pairs": 0.0,
                    "train/count_optimizer_steps_per_training_iteration": 0.0}
        model.train()
        model.config.use_cache = False
        device = next(model.parameters()).device
        pairs = torch.tensor(pairs, device=rollout.input_ids.device)
        rng = torch.Generator(device=pairs.device).manual_seed(self._next_update_seed())
        params = [p for p in model.parameters() if p.requires_grad]
        accumulation = max(1, grad_accum_steps)
        total_loss = 0.0
        batches = pending = steps = 0
        optimizer.zero_grad(set_to_none=True)

        def step(count):
            if count < accumulation:
                for param in params:
                    if param.grad is not None:
                        param.grad.mul_(accumulation / count)
            norm = clip_grad_norm_(params, self.cfg.max_grad_norm)
            valid = math.isfinite(norm)
            if valid:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            return int(valid)

        for _ in range(self.cfg.ppo_epochs):
            order = torch.randperm(len(pairs), generator=rng, device=pairs.device)
            for start in range(0, len(pairs), self.cfg.minibatch_size):
                selected = pairs[order[start:start + self.cfg.minibatch_size]]
                idx = torch.cat((selected[:, 0], selected[:, 1]))
                ids = rollout.input_ids.index_select(0, idx).to(device)
                attention = rollout.attention_mask.index_select(0, idx).to(device)
                mask = rollout.completion_mask.index_select(0, idx).to(device)
                new = compute_per_token_logprobs(model, ids, attention)
                ref = rollout.ref_logprobs.index_select(0, idx).to(device).detach()
                chosen, rejected = ((new - ref) * mask).sum(dim=1).chunk(2)
                loss = -F.logsigmoid(self.beta * (chosen - rejected)).mean()
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
                    continue
                (loss / accumulation).backward()
                pending += 1
                total_loss += loss.detach().item()
                batches += 1
                if pending == accumulation:
                    steps += step(pending)
                    pending = 0
        if pending:
            steps += step(pending)
        return {"train/online_dpo_loss": total_loss / max(1, batches),
                "train/online_dpo_pairs": float(len(pairs)),
                "train/count_optimizer_steps_per_training_iteration": float(steps)}
################################
