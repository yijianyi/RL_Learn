from __future__ import annotations

import torch
import torch.nn.functional as F

from llm_rl_final_proj.models.load import PolicyModel


def compute_per_token_logprobs(
    model: PolicyModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    enable_grad: bool = True,
) -> torch.Tensor:
    """Returns log p(x_t | x_<t) for t in [1, L-1]. Shape: [B, L-1]."""
    with torch.set_grad_enabled(enable_grad):
        ################################
        # answer code
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        logits = logits[:, :-1, :].float()
        targets = input_ids[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              targets.reshape(-1), reduction="none")
        return -nll.reshape_as(targets)
        ################################


def build_completion_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_input_len: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Mask over per-token positions [B, L-1], selecting completion tokens only."""
    del pad_token_id
    # TODO(student): build a float mask of shape [B, L-1] that selects only completion tokens.
    # Be careful about the one-token shift between logits[:, :-1] and input_ids[:, 1:].
    ################################
    # answer code
    positions = torch.arange(1, input_ids.shape[1], device=input_ids.device)
    return ((positions >= prompt_input_len)[None] & attention_mask[:, 1:].bool()).float()
    ################################


def masked_sum(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


def masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def approx_kl_from_logprobs(
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
    log_ratio_clip: float = 20.0,
) -> torch.Tensor:
    """Positive KL proxy from sampled actions.

    Uses estimator: exp(delta) - delta - 1 where delta = log p_ref(a) - log p_new(a).
    """
    # Keep eps and the clipping bound for the stable masked KL estimate.
    # TODO(student): implement the sampled-token KL proxy used throughout the codebase.
    # You should mask out non-completion positions and return a scalar batch mean.
    ################################
    # answer code
    delta = (ref_logprobs - new_logprobs).clamp(-log_ratio_clip, log_ratio_clip)
    per_token = torch.expm1(delta) - delta
    return (per_token * mask).sum() / (mask.sum() + eps)
    ################################
