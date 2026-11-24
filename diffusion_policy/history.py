"""Rolling window of the last K (s, a, r, s') tuples for each environment.

Tuples are right-aligned: position K-1 holds the most recent one. Positions
that have not been filled yet are masked out.
"""

import torch


def tuple_dim(obs_dim, act_dim):
    return 2 * obs_dim + act_dim + 1


class History:
    def __init__(self, num_envs, K, dim, device="cpu"):
        self.tuples = torch.zeros(num_envs, K, dim, device=device)
        self.mask = torch.zeros(num_envs, K, dtype=torch.bool, device=device)

    def push(self, obs, action, reward, next_obs):
        new = torch.cat([obs, action, reward.unsqueeze(-1), next_obs], dim=-1)
        self.tuples = torch.cat([self.tuples[:, 1:], new.unsqueeze(1)], dim=1)
        self.mask = torch.cat([self.mask[:, 1:], torch.ones_like(self.mask[:, :1])], dim=1)

    def clear(self):
        self.tuples = torch.zeros_like(self.tuples)
        self.mask = torch.zeros_like(self.mask)

    def snapshot(self):
        return self.tuples.clone(), self.mask.clone()
