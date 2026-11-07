from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class IFQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor_flow,
        make_actor_flow_optimizer,
        make_critic,
        make_critic_optimizer,
        make_value,
        make_value_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        num_samples: int = 32,
        expectile: float = 0.9,
    ):
        super().__init__()

        self.action_dim = action_dim
        
        ################################
        # answer code
        self.actor_flow = make_actor_flow(observation_shape, action_dim)
        ################################

        ################################
        # answer code
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.value = make_value(observation_shape)
        ################################

        ################################
        # answer code
        self.actor_flow_optimizer = make_actor_flow_optimizer(self.actor_flow.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.value_optimizer = make_value_optimizer(self.value.parameters())
        if flow_steps < 1 or num_samples < 1:
            raise ValueError("flow_steps and num_samples must be positive")
        ################################

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.num_samples = num_samples
        self.expectile = expectile

    @staticmethod
    def expectile_loss(adv: torch.Tensor, expectile: float) -> torch.Tensor:
        """
        Compute the expectile loss for IFQL
        """
        ################################
        # answer code
        return torch.where(adv >= 0, expectile, 1 - expectile) * adv.square()
        ################################

    @torch.compile
    def update_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """
        Update value function
        """
        ################################
        # answer code
        with torch.no_grad():
            q = self.target_critic(observations, actions).min(dim=0).values
        v = self.value(observations)
        loss = self.expectile_loss(q - v, self.expectile).mean()
        self.value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.value_optimizer.step()
        return {"v_loss": loss.detach(), "v_mean": v.detach().mean()}
        ################################

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Rejection / best-of-n sampling using the flow policy and critic.

        We:
          1. Sample multiple candidate actions via the BC flow.
          2. Evaluate them with the critic.
          3. Pick the action with the highest Q-value.
        """
        ################################
        # answer code
        batch = observations.shape[0]
        expanded = observations.repeat_interleave(self.num_samples, dim=0)
        noise = torch.randn((expanded.shape[0], self.action_dim), device=observations.device)
        candidates = self.get_flow_action(expanded, noise)
        scores = self.critic(expanded, candidates).min(dim=0).values.reshape(batch, self.num_samples)
        best = scores.argmax(dim=1)
        return candidates.reshape(batch, self.num_samples, self.action_dim)[
            torch.arange(batch, device=observations.device), best]
        ################################

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        ################################
        # answer code
        obs = ptu.from_numpy(np.asarray(observation))[None]
        return ptu.to_numpy(self.sample_actions(obs)[0])
        ################################

    @torch.compile
    def get_flow_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Compute the flow action using Euler integration for `self.flow_steps` steps.
        """
        ################################
        # answer code
        action = noise
        for i in range(self.flow_steps):
            times = action.new_full((*action.shape[:-1], 1), i / self.flow_steps)
            action = action + self.actor_flow(observation, action, times) / self.flow_steps
        return action.clamp(-1, 1)
        ################################

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a) using the learned value function for bootstrapping,
        as in IFQL / IQL-style critic training.
        """
        ################################
        # answer code
        with torch.no_grad():
            target = rewards + self.discount * (1 - dones.float()) * self.value(next_observations)
        q = self.critic(observations, actions)
        loss = (q - target[None]).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        return {"q_loss": loss.detach(), "q_mean": q.detach().mean()}
        ################################


    @torch.compile
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the flow actor using the velocity matching loss.
        """
        ################################
        # answer code
        noise = torch.randn_like(actions)
        times = torch.rand((*actions.shape[:-1], 1), device=actions.device)
        x_t = (1 - times) * noise + times * actions
        loss = (self.actor_flow(observations, x_t, times) - (actions - noise)).square().mean()
        self.actor_flow_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_flow_optimizer.step()
        return {"loss": loss.detach()}
        ################################


    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_v = self.update_value(observations, actions)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"value/{k}": v.item() for k, v in metrics_v.items()},
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        ################################
        # answer code
        with torch.no_grad():
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.lerp_(source, self.target_update_rate)
        ################################
