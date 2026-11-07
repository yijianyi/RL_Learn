from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:
    ################################
    # answer code
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    n = batch.input_ids.shape[0]
    index_device = generator.device if generator is not None else batch.input_ids.device
    indices = (torch.randperm(n, generator=generator, device=index_device) if shuffle
               else torch.arange(n, device=index_device))
    tensor_fields = ("input_ids", "attention_mask", "completion_mask", "old_logprobs",
                     "ref_logprobs", "rewards", "advantages")
    for start in range(0, n, minibatch_size):
        idx = indices[start:start + minibatch_size]
        fields = {name: getattr(batch, name).index_select(
            0, idx.to(getattr(batch, name).device)) for name in tensor_fields}
        for name in ("task_names", "completion_texts"):
            values = getattr(batch, name)
            fields[name] = None if values is None else [values[i] for i in idx.tolist()]
        mb = RolloutBatch(**fields)
        yield mb.to(device) if device is not None else mb
    ################################
