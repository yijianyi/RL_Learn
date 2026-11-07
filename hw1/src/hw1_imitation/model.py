"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        ################################
        # answer code
        dims = (state_dim, *hidden_dims, chunk_size * action_dim)
        layers = []
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(din, dout))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        ################################

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        ################################
        # answer code
        return nn.functional.mse_loss(self.sample_actions(state), action_chunk)
        ################################

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        ################################
        # answer code
        return self.network(state).reshape(-1, self.chunk_size, self.action_dim)
        ################################


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        ################################
        # answer code
        flat_dim = chunk_size * action_dim
        dims = (state_dim + flat_dim + 1, *hidden_dims, flat_dim)
        layers = []
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(din, dout))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.velocity = nn.Sequential(*layers)
        ################################

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        ################################
        # answer code
        target = action_chunk.flatten(1)
        noise = torch.randn_like(target)
        t = torch.rand((state.shape[0], 1), device=state.device, dtype=state.dtype)
        x_t = (1 - t) * noise + t * target
        prediction = self.velocity(torch.cat((state, x_t, t), dim=-1))
        return nn.functional.mse_loss(prediction, target - noise)
        ################################

    ################################
    # answer code
    @torch.no_grad()
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        x = torch.randn((state.shape[0], self.chunk_size * self.action_dim),
                        device=state.device, dtype=state.dtype)
        for i in range(num_steps):
            t = state.new_full((state.shape[0], 1), i / num_steps)
            x = x + self.velocity(torch.cat((state, x, t), dim=-1)) / num_steps
        return x.reshape(-1, self.chunk_size, self.action_dim)
    ################################


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
