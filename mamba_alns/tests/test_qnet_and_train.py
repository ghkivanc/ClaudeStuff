import copy
import math
import random

import numpy as np
import torch

from mamba_alns.cvrp import ACTIONS, random_instance, solution_cost, initial_solution
from mamba_alns.qnet import MambaQNet
from mamba_alns.replay import ReplayBuffer, stack_windows
from mamba_alns.train import Config, run_episode, train, train_step


def small_cfg():
    return Config(n_customers=10, capacity=20, iterations=12, episodes=3, window=4,
                  n_step=3, batch_size=4, updates_per_episode=2, min_transitions=1,
                  target_sync=2, device="cpu")


def small_qnet():
    return MambaQNet(len(ACTIONS), gnn_hidden=16, gnn_layers=1, gnn_heads=2,
                     d_model=16, mamba_layers=1)


def test_qnet_output_shape_and_grads():
    cfg = small_cfg()
    qnet = small_qnet()
    inst = random_instance(cfg.n_customers, cfg.capacity, np.random.default_rng(0))
    ep, _ = run_episode(qnet, inst, cfg, 1.0, np.random.default_rng(0))
    q = qnet(*stack_windows([ep.window(0, 4), ep.window(7, 4), ep.window(12, 4)]))
    assert q.shape == (3, len(ACTIONS))
    q.sum().backward()
    assert qnet.gnn.input.weight.grad is not None   # GNN trained end-to-end


def test_run_episode():
    cfg = small_cfg()
    rng = np.random.default_rng(1)
    inst = random_instance(cfg.n_customers, cfg.capacity, rng)
    start = solution_cost(initial_solution(inst), inst)
    ep, best = run_episode(small_qnet(), inst, cfg, 0.5, rng)
    assert len(ep) == cfg.iterations
    assert best <= start + 1e-9
    assert all(0 < r <= 1 for r in ep.rewards)
    assert all(0 <= a < len(ACTIONS) for a in ep.actions)


def test_train_step_reduces_loss_on_fixed_buffer():
    torch.manual_seed(0)
    cfg = small_cfg()
    cfg.lr = 1e-3
    qnet = small_qnet()
    target = copy.deepcopy(qnet)
    opt = torch.optim.Adam(qnet.parameters(), lr=cfg.lr)
    buf = ReplayBuffer(5)
    inst = random_instance(cfg.n_customers, cfg.capacity, np.random.default_rng(2))
    buf.add(run_episode(qnet, inst, cfg, 1.0, np.random.default_rng(2))[0])

    random.seed(0)
    first = [train_step(qnet, target, opt, buf, cfg) for _ in range(5)]
    for _ in range(50):
        train_step(qnet, target, opt, buf, cfg)
    last = [train_step(qnet, target, opt, buf, cfg) for _ in range(5)]
    assert all(math.isfinite(l) for l in first + last)
    assert sum(last) < sum(first)


def test_train_smoke():
    qnet = train(small_cfg())
    assert isinstance(qnet, MambaQNet)
