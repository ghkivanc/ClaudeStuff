"""Episode storage, K-step windows, N-step returns and the replay buffer.

An episode of T ALNS iterations stores
  states  s_0 .. s_T   (as edge_index tensors; node features are per instance)
  actions a_0 .. a_{T-1}
  rewards r_0 .. r_{T-1}
A transition is just (episode, t), so windows and N-step targets are built
when sampling and no data is duplicated.
"""
import random
from collections import deque

import torch

from .gnn import node_features, solution_edges


class Episode:
    def __init__(self, inst, initial_routes):
        self.node_feats = node_features(inst)
        self.dist = torch.as_tensor(inst.dist, dtype=torch.float32)
        self.n_nodes = len(inst.demands)
        self.edges = [solution_edges(initial_routes, self.n_nodes)]
        self.actions = []
        self.rewards = []

    def add_step(self, action, reward, next_routes):
        self.actions.append(action)
        self.rewards.append(reward)
        self.edges.append(solution_edges(next_routes, self.n_nodes))

    def __len__(self):
        return len(self.actions)

    def window(self, t, k):
        """The K tokens ending at state s_t.

        Steps before 0 are padded by repeating step 0. Returns
        (graphs, prev_actions, prev_rewards) for one window, where for step j
        the previous action/reward are a_{j-1}, r_{j-1} (-1 / 0.0 at j = 0).
        """
        steps = [max(0, t - k + 1 + i) for i in range(k)]
        graphs = [(self.node_feats, self.edges[j], self.dist) for j in steps]
        prev_actions = [self.actions[j - 1] if j > 0 else -1 for j in steps]
        prev_rewards = [self.rewards[j - 1] if j > 0 else 0.0 for j in steps]
        return graphs, prev_actions, prev_rewards


def stack_windows(windows):
    """List of windows -> (graphs, prev_actions (B,K), prev_rewards (B,K)) for MambaQNet."""
    graphs = [g for w in windows for g in w[0]]
    prev_actions = torch.tensor([w[1] for w in windows], dtype=torch.long)
    prev_rewards = torch.tensor([w[2] for w in windows], dtype=torch.float32)
    return graphs, prev_actions, prev_rewards


def n_step_return(rewards, t, n, gamma):
    """G = r_t + g*r_{t+1} + ... + g^(n-1)*r_{t+n-1}, cut short at the episode end.

    Returns (G, bootstrap). bootstrap is True when s_{t+n} is not the final
    state, i.e. the target should add g^n * max_a Q(s_{t+n}, a).
    """
    T = len(rewards)
    steps = min(n, T - t)
    G = sum(gamma ** i * rewards[t + i] for i in range(steps))
    return G, t + n < T


class ReplayBuffer:
    def __init__(self, max_episodes):
        self.episodes = deque(maxlen=max_episodes)

    def add(self, episode):
        self.episodes.append(episode)

    def n_transitions(self):
        return sum(len(ep) for ep in self.episodes)

    def sample(self, batch_size):
        """Uniform (episode, t) pairs."""
        out = []
        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            out.append((ep, random.randrange(len(ep))))
        return out
