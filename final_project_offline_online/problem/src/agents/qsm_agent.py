from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List

class QSMAgent(nn.Module):
    ################################
    # answer code
    """BC-regularized Q-score matching with an epsilon-prediction DDPM."""
    ################################

    ################################
    # answer code
    def __init__(self, observation_shape, action_dim, make_actor, make_actor_optimizer,
                 make_critic, make_critic_optimizer, discount, target_update_rate,
                 alpha, inv_temp, flow_steps):
        super().__init__()
        if flow_steps < 1:
            raise ValueError("flow_steps must be positive")
        self.action_dim = action_dim
        self.actor = make_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.discount, self.target_update_rate = discount, target_update_rate
        self.alpha, self.inv_temp, self.flow_steps = alpha, inv_temp, flow_steps
        betas = self.cosine_beta_schedule(flow_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alpha_hats", (1 - betas).cumprod(dim=0))
        self.to(ptu.device)
    ################################

    ################################
    # answer code
    def cosine_beta_schedule(self, timesteps):
        if timesteps < 1:
            raise ValueError("timesteps must be positive")
        t = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
        cumulative = torch.cos((t + 0.008) / 1.008 * np.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        return (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-5, 0.999).float()
    ################################

    ################################
    # answer code
    @torch.compiler.disable
    @torch.no_grad()
    def ddpm_sampler(self, observations, noise):
        x = noise
        for i in reversed(range(self.flow_steps)):
            times = x.new_full((*x.shape[:-1], 1), i / self.flow_steps)
            eps = self.actor(observations, x, times)
            a, abar, beta = self.alphas[i], self.alpha_hats[i], self.betas[i]
            # Clip the estimated clean action, then use the DDPM posterior mean.
            x0 = ((x - (1 - abar).sqrt() * eps) / abar.sqrt()).clamp(-1, 1)
            previous_abar = self.alpha_hats[i - 1] if i > 0 else abar.new_tensor(1.0)
            mean = (previous_abar.sqrt() * beta / (1 - abar) * x0
                    + a.sqrt() * (1 - previous_abar) / (1 - abar) * x)
            variance = beta * (1 - previous_abar) / (1 - abar)
            x = mean + variance.sqrt() * torch.randn_like(x) if i > 0 else mean
        return x.clamp(-1, 1)
    ################################

    ################################
    # answer code
    @torch.no_grad()
    def get_action(self, observation):
        obs = ptu.from_numpy(np.asarray(observation))[None]
        noise = torch.randn((1, self.action_dim), device=obs.device)
        return ptu.to_numpy(self.ddpm_sampler(obs, noise)[0])
    ################################

    # Sampling uses a Python denoising loop, so this update runs eagerly.
    ################################
    # answer code
    @torch.compiler.disable
    def update_q(self, observations, actions, rewards, next_observations, dones):
        with torch.no_grad():
            next_actions = self.ddpm_sampler(next_observations, torch.randn_like(actions))
            next_q = self.target_critic(next_observations, next_actions).min(dim=0).values
            target = rewards + self.discount * (1 - dones.float()) * next_q
        q = self.critic(observations, actions)
        loss = (q - target[None]).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        return {"q_loss": loss.detach(), "q_mean": q.detach().mean()}
    ################################

    ################################
    # answer code
    @torch.compiler.disable
    def update_actor(self, observations, actions):
        indices = torch.randint(self.flow_steps, (actions.shape[0],), device=actions.device)
        abar = self.alpha_hats[indices, None]
        noise = torch.randn_like(actions)
        noisy_actions = (abar.sqrt() * actions + (1 - abar).sqrt() * noise).detach().requires_grad_(True)
        # Epsilon points opposite to the score. Stop gradients through the critic target.
        q = self.critic(observations, noisy_actions).mean(dim=0)
        q_gradient = torch.autograd.grad(q.sum(), noisy_actions)[0].detach()
        times = indices[:, None].to(actions.dtype) / self.flow_steps
        eps = self.actor(observations, noisy_actions.detach(), times)
        qsm_loss = (eps + self.inv_temp * q_gradient).square().mean()
        bc_loss = (eps - noise).square().mean()
        loss = qsm_loss + self.alpha * bc_loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_optimizer.step()
        return {"loss": loss.detach(), "qsm_loss": qsm_loss.detach(), "bc_loss": bc_loss.detach()}
    ################################

    ################################
    # answer code
    def update(self, observations, actions, rewards, next_observations, dones, step):
        q = self.update_q(observations, actions, rewards, next_observations, dones)
        actor = self.update_actor(observations, actions)
        self.update_target_critic()
        return {**{f"critic/{k}": v.item() for k, v in q.items()},
                **{f"actor/{k}": v.item() for k, v in actor.items()}}
    ################################

    ################################
    # answer code
    @torch.no_grad()
    def update_target_critic(self):
        for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.lerp_(source, self.target_update_rate)
    ################################
