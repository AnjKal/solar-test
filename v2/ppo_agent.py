import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, kl_divergence
from typing import Optional


def _mask_logits(logits: torch.Tensor, action_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mask invalid actions by setting their logits to a very negative value.

    action_mask must be broadcastable to logits shape and contain 0/1 (or bool) values.
    """

    if action_mask is None:
        return logits

    if action_mask.dtype != torch.bool:
        action_mask = action_mask.to(dtype=torch.bool)

    # Invalid actions -> extremely low logit (approx -inf) so probability ~ 0.
    return logits.masked_fill(~action_mask, -1e9)
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.policy(features)
        value = self.value(features)
        return logits, value
class PPOAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    def select_action(self, obs, action_mask=None):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        logits, value = self.model(obs)
        if action_mask is not None:
            mask_t = torch.tensor(action_mask, device=self.device)
            logits = _mask_logits(logits, mask_t)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.detach(), value.detach()
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0

        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return advantages
    def update(
        self,
        obs,
        actions,
        old_log_probs,
        returns,
        advantages,
        action_masks=None,
        epochs=4,
        batch_size=256,
    ):

        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        masks_t = None
        if action_masks is not None:
            masks_t = torch.tensor(action_masks, dtype=torch.bool).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Snapshot policy logits before any update steps so we can report
        # KL(π_t || π_{t-1}) where π_{t-1} is the pre-update policy.
        with torch.no_grad():
            old_logits_all, _ = self.model(obs)
            old_logits_all = _mask_logits(old_logits_all, masks_t)
            old_logits_all = old_logits_all.detach()

        metrics = {
            "kl_div": 0.0,
            "policy_entropy": 0.0,
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
            "n_minibatches": 0,
        }

        for _ in range(epochs):
            idx = np.random.permutation(len(obs))
            for i in range(0, len(obs), batch_size):
                batch = idx[i:i+batch_size]

                logits, values = self.model(obs[batch])
                batch_masks = masks_t[batch] if masks_t is not None else None
                logits = _mask_logits(logits, batch_masks)
                dist = Categorical(logits=logits)
                old_dist = Categorical(logits=old_logits_all[batch])

                log_probs = dist.log_prob(actions[batch])
                entropy = dist.entropy().mean()

                # KL divergence between current and pre-update policies.
                # Averaged over the minibatch: KL(π_t || π_{t-1}).
                kl = kl_divergence(dist, old_dist).mean()

                ratio = torch.exp(log_probs - old_log_probs[batch])

                surr1 = ratio * advantages[batch]
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_eps,
                    1 + self.clip_eps
                ) * advantages[batch]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values.squeeze(), returns[batch])

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # Metrics accumulation (no effect on gradients)
                metrics["kl_div"] += float(kl.detach().item())
                metrics["policy_entropy"] += float(entropy.detach().item())
                metrics["value_loss"] += float(value_loss.detach().item())
                metrics["policy_loss"] += float(policy_loss.detach().item())
                metrics["entropy_loss"] += float((-self.ent_coef * entropy).detach().item())
                metrics["total_loss"] += float(loss.detach().item())
                metrics["n_minibatches"] += 1

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        n = max(1, int(metrics["n_minibatches"]))
        return {
            "kl_div": metrics["kl_div"] / n,
            "policy_entropy": metrics["policy_entropy"] / n,
            "value_loss": metrics["value_loss"] / n,
            "policy_loss": metrics["policy_loss"] / n,
            "entropy_loss": metrics["entropy_loss"] / n,
            "total_loss": metrics["total_loss"] / n,
        }
