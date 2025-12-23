import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
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
    def select_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        logits, value = self.model(obs)
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
    def update(self, obs, actions, old_log_probs, returns, advantages, epochs=4, batch_size=256):

        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            idx = np.random.permutation(len(obs))
            for i in range(0, len(obs), batch_size):
                batch = idx[i:i+batch_size]

                logits, values = self.model(obs[batch])
                dist = Categorical(logits=logits)

                log_probs = dist.log_prob(actions[batch])
                entropy = dist.entropy().mean()

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

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
