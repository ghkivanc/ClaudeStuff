"""N-step Q-learning of ALNS operator selection with a GAT + Mamba Q-network.

Each episode: a fresh random CVRP instance, run `iterations` ALNS steps.
  - action    = (destroy, repair) pair, picked epsilon-greedily from Q
  - acceptance: always accept the new solution (best_so_far tracked separately)
  - reward    = e^(best_so_far - current_cost)
Target for transition (s_t, a_t):
  G = sum_{i<N} gamma^i r_{t+i}  +  gamma^N * max_a Q_target(s_{t+N}, a)
  (bootstrap term dropped when s_{t+N} is the last state of the episode)

Run:  python -m mamba_alns.train
"""
import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .cvrp import ACTIONS, apply_action, initial_solution, random_instance, reward, solution_cost
from .qnet import MambaQNet
from .replay import Episode, ReplayBuffer, n_step_return, stack_windows


@dataclass
class Config:
    # problem / ALNS
    n_customers: int = 50
    capacity: int = 40
    destroy_fraction: float = 0.2     # share of customers removed per destroy
    iterations: int = 100             # ALNS iterations per episode (T)
    # Q-learning
    episodes: int = 300
    window: int = 8                   # K: tokens Mamba sees
    n_step: int = 5                   # N: steps in the contraction
    gamma: float = 0.95
    lr: float = 1e-4
    batch_size: int = 32
    buffer_episodes: int = 200
    updates_per_episode: int = 25     # gradient steps after each episode
    min_transitions: int = 1000       # start learning after this many stored
    target_sync: int = 250            # gradient steps between target-net copies
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 200
    # misc
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def epsilon(cfg, episode):
    frac = min(1.0, episode / cfg.eps_decay_episodes)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def select_action(qnet, episode, t, cfg, eps):
    if random.random() < eps:
        return random.randrange(len(ACTIONS))
    with torch.no_grad():
        q = qnet(*stack_windows([episode.window(t, cfg.window)]))
    return int(q.argmax(dim=1).item())


def run_episode(qnet, inst, cfg, eps, rng):
    """Run one ALNS episode on inst. Returns the Episode and the best cost found."""
    routes = initial_solution(inst)
    best = solution_cost(routes, inst)
    episode = Episode(inst, routes)
    k = max(1, int(cfg.destroy_fraction * inst.n_customers))

    for t in range(cfg.iterations):
        action = select_action(qnet, episode, t, cfg, eps)
        routes = apply_action(action, routes, inst, k, rng)   # always accept
        cost = solution_cost(routes, inst)
        best = min(best, cost)
        episode.add_step(action, reward(cost, best), routes)
    return episode, best


def train_step(qnet, target_net, optimizer, buffer, cfg):
    batch = buffer.sample(cfg.batch_size)

    windows, next_windows, actions, returns, bootstrap = [], [], [], [], []
    for ep, t in batch:
        G, boot = n_step_return(ep.rewards, t, cfg.n_step, cfg.gamma)
        windows.append(ep.window(t, cfg.window))
        next_windows.append(ep.window(min(t + cfg.n_step, len(ep)), cfg.window))
        actions.append(ep.actions[t])
        returns.append(G)
        bootstrap.append(float(boot))

    device = cfg.device
    actions = torch.tensor(actions, device=device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    bootstrap = torch.tensor(bootstrap, device=device)

    q = qnet(*stack_windows(windows)).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(*stack_windows(next_windows)).max(dim=1).values
    target = returns + cfg.gamma ** cfg.n_step * next_q * bootstrap

    loss = F.smooth_l1_loss(q, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
    optimizer.step()
    return loss.item()


def train(cfg):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    qnet = MambaQNet(len(ACTIONS)).to(cfg.device)
    target_net = copy.deepcopy(qnet)
    optimizer = torch.optim.Adam(qnet.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_episodes)
    grad_steps = 0

    for ep_idx in range(cfg.episodes):
        eps = epsilon(cfg, ep_idx)
        inst = random_instance(cfg.n_customers, cfg.capacity, rng)
        episode, best = run_episode(qnet, inst, cfg, eps, rng)
        buffer.add(episode)

        losses = []
        if buffer.n_transitions() >= cfg.min_transitions:
            for _ in range(cfg.updates_per_episode):
                losses.append(train_step(qnet, target_net, optimizer, buffer, cfg))
                grad_steps += 1
                if grad_steps % cfg.target_sync == 0:
                    target_net.load_state_dict(qnet.state_dict())

        mean_loss = sum(losses) / len(losses) if losses else float("nan")
        print(f"episode {ep_idx:4d}  eps {eps:.2f}  best {best:.3f}  loss {mean_loss:.4f}")
    return qnet


def evaluate(qnet, cfg, n_instances=20, seed=12345):
    """Learned greedy policy vs. uniformly random operator choice on the same instances."""
    rng = np.random.default_rng(seed)
    learned, rand = [], []
    for _ in range(n_instances):
        inst = random_instance(cfg.n_customers, cfg.capacity, rng)
        op_seed = int(rng.integers(1 << 31))
        _, best_learned = run_episode(qnet, inst, cfg, 0.0, np.random.default_rng(op_seed))
        _, best_random = run_episode(qnet, inst, cfg, 1.0, np.random.default_rng(op_seed))
        learned.append(best_learned)
        rand.append(best_random)
    print(f"mean best cost  learned: {np.mean(learned):.3f}   random: {np.mean(rand):.3f}")
    return float(np.mean(learned)), float(np.mean(rand))


if __name__ == "__main__":
    cfg = Config()
    qnet = train(cfg)
    torch.save(qnet.state_dict(), "mamba_qnet.pt")
    evaluate(qnet, cfg)
