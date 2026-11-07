from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Sequence


class DSRLAgent(nn.Module):
    ################################
    # answer code
    """Noise-aliased DSRL: https://arxiv.org/abs/2506.15799.

    Pretrain the BC flow offline, then freeze it while steering its input noise online.
    """
    ################################

    ################################
    # answer code
    def __init__(self, observation_shape, action_dim,
                 make_bc_flow_actor, make_bc_flow_actor_optimizer,
                 make_noise_actor, make_noise_actor_optimizer,
                 make_critic, make_critic_optimizer,
                 make_z_critic, make_z_critic_optimizer,
                 discount, target_update_rate, flow_steps, noise_scale=1.0,
                 online_training=False):
        super().__init__()
        if flow_steps < 1 or noise_scale <= 0:
            raise ValueError("flow_steps and noise_scale must be positive")
        self.action_dim, self.flow_steps = action_dim, flow_steps
        self.discount, self.target_update_rate = discount, target_update_rate
        self.noise_scale, self.target_entropy = noise_scale, -float(action_dim)
        self.online_training = online_training
        self.bc_flow_actor = make_bc_flow_actor(observation_shape, action_dim)
        self.target_bc_flow_actor = make_bc_flow_actor(observation_shape, action_dim)
        self.target_bc_flow_actor.load_state_dict(self.bc_flow_actor.state_dict())
        self.noise_actor = make_noise_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.z_critic = make_z_critic(observation_shape, action_dim)
        self.log_alpha = nn.Parameter(torch.zeros((), device=ptu.device))
        self.bc_flow_actor_optimizer = make_bc_flow_actor_optimizer(self.bc_flow_actor.parameters())
        self.noise_actor_optimizer = make_noise_actor_optimizer(self.noise_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.z_critic_optimizer = make_z_critic_optimizer(self.z_critic.parameters())
        self.alpha_optimizer = make_noise_actor_optimizer([self.log_alpha])
        self.to(ptu.device)
    ################################

    @property
    def alpha(self):
        ################################
        # answer code
        return self.log_alpha.exp()
        ################################

    ################################
    # answer code
    @torch.compiler.disable
    @torch.no_grad()
    def sample_flow_actions(self, observations, noises):
        action = noises
        for i in range(self.flow_steps):
            times = action.new_full((*action.shape[:-1], 1), i / self.flow_steps)
            action = action + self.target_bc_flow_actor(observations, action, times) / self.flow_steps
        return action.clamp(-1, 1)
    ################################

    ################################
    # answer code
    @torch.no_grad()
    def sample_actions(self, observations):
        noise = self.noise_actor(observations).sample() * self.noise_scale
        return self.sample_flow_actions(observations, noise)
    ################################

    ################################
    # answer code
    @torch.no_grad()
    def get_action(self, observation):
        obs = ptu.from_numpy(np.asarray(observation))[None]
        return ptu.to_numpy(self.sample_actions(obs)[0])
    ################################

    ################################
    # answer code
    def update_q(self, observations, actions, rewards, next_observations, dones):
        with torch.no_grad():
            next_actions = self.sample_actions(next_observations)
            next_q = self.target_critic(next_observations, next_actions).min(dim=0).values
            target = rewards + self.discount * (1 - dones.float()) * next_q
        q = self.critic(observations, actions)
        loss = (q - target[None]).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        return {"loss": loss.detach(), "q_mean": q.detach().mean()}
    ################################

    ################################
    # answer code
    def update_qz(self, observations, actions, noises):
        with torch.no_grad():
            targets = self.critic(observations, actions).min(dim=0).values
        qz = self.z_critic(observations, noises)
        loss = (qz - targets[None]).square().mean()
        self.z_critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.z_critic_optimizer.step()
        return {"loss": loss.detach()}
    ################################

    ################################
    # answer code
    def update_actor(self, observations, actions):
        noise = torch.randn_like(actions)
        times = torch.rand((*actions.shape[:-1], 1), device=actions.device)
        x_t = (1 - times) * noise + times * actions
        loss = (self.bc_flow_actor(observations, x_t, times) - (actions - noise)).square().mean()
        self.bc_flow_actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.bc_flow_actor_optimizer.step()
        return {"loss": loss.detach()}
    ################################

    ################################
    # answer code
    def update_noise_actor(self, observations):
        dist = self.noise_actor(observations)
        raw_noise = dist.rsample()
        noises = raw_noise * self.noise_scale
        # Include the change-of-variables constant for the scaled noise policy.
        log_prob = dist.log_prob(raw_noise) - self.action_dim * np.log(self.noise_scale)
        qz = self.z_critic(observations, noises).min(dim=0).values
        loss = (self.alpha.detach() * log_prob - qz).mean()
        self.noise_actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.noise_actor_optimizer.step()
        self._noise_log_probs = log_prob.detach()
        return {"loss": loss.detach(), "entropy": -log_prob.detach().mean()}
    ################################

    ################################
    # answer code
    def update_alpha(self):
        loss = -(self.log_alpha * (self._noise_log_probs + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.alpha_optimizer.step()
        return {"loss": loss.detach(), "alpha": self.alpha.detach()}
    ################################

    ################################
    # answer code
    @torch.no_grad()
    def update_target_critic(self):
        for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.lerp_(source, self.target_update_rate)
    ################################

    ################################
    # answer code
    @torch.no_grad()
    def update_target_bc_flow_actor(self):
        for target, source in zip(self.target_bc_flow_actor.parameters(), self.bc_flow_actor.parameters()):
            target.lerp_(source, self.target_update_rate)
    ################################

    ################################
    # answer code
    def update(self, observations, actions, rewards, next_observations, dones, step):
        metrics = {}
        if not self.online_training:
            metrics['actor/loss'] = self.update_actor(observations, actions)['loss'].item()
            self.update_target_bc_flow_actor()
        q = self.update_q(observations, actions, rewards, next_observations, dones)
        # Qz must see matching (noise, generated action) pairs, never unrelated dataset actions.
        noises = torch.randn_like(actions)
        generated = self.sample_flow_actions(observations, noises)
        qz = self.update_qz(observations, generated, noises)
        actor = self.update_noise_actor(observations)
        alpha = self.update_alpha()
        self.update_target_critic()
        for prefix, values in (("critic", q), ("noise_critic", qz), ("noise_actor", actor), ("alpha", alpha)):
            metrics.update({f"{prefix}/{k}": v.item() for k, v in values.items()})
        return metrics
    ################################

