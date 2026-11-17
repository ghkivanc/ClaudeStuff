# Mamba Q-learning for ALNS operator selection (CVRP)

Adaptive Large Neighbourhood Search (ALNS) improves a vehicle-routing solution by
repeatedly destroying part of it and repairing it. This project learns **which
(destroy, repair) pair to use at each iteration** with N-step Q-learning. The
Q-network is a graph attention network (GAT) that encodes each solution,
followed by a Mamba sequence model over the recent history.

## How it works

One **episode** is one ALNS run of `T` iterations on a fresh random CVRP instance.

```
s_0 = nearest-neighbour solution,   ref = LP lower bound of the instance
for t = 0 .. T-1:
    a_t     = epsilon-greedy over Q(window ending at s_t)     # 9 actions
    s_{t+1} = repair(destroy(s_t))                             # always accepted
    r_t     = 1 / ((cost(s_{t+1}) - ref) / ref + eps)
```

| Piece | Choice |
|---|---|
| Problem | CVRP, uniform random instances in the unit square (default 50 customers, capacity 40, demands 1–9) |
| Actions | 3 destroy × 3 repair = 9 pairs. Destroy (removes 20% of customers): random / worst / related. Repair: greedy / regret-2 / random insertion |
| Acceptance | the new solution is always accepted; the best solution found is tracked separately |
| Exploration | epsilon-greedy, epsilon decays linearly from 1.0 to 0.05 |
| Reward | `1 / ((current - ref) / ref + eps)`, eps = 1e-3. `ref` is the LP relaxation of the single-commodity flow (Gavish–Graves) model, solved once per instance with scipy/HiGHS |
| Solution encoder | GAT over the route arcs (both directions) + self-loops. Node features `[x, y, demand/capacity, is_depot]`, edge feature = distance, mean pooled |
| Sequence model | pure-PyTorch Mamba over the last K = 8 tokens |
| Token at step j | `[GAT(s_j), one_hot(a_{j-1}), r_{j-1}]` |
| Q-values | linear head on the last token's output |
| Target | `Σ_{i<N} γ^i r_{t+i} + γ^N · max_a Q_target(s_{t+N}, a)`, N = 5. No bootstrap when `s_{t+N}` is the episode's last state |
| Learning | replay buffer of whole episodes, Huber loss, Adam, target network copied every 250 gradient steps. The GAT is trained end-to-end through the Q-loss |

## Setup

Requires Python 3.10+.

```bash
git clone <this repo> && cd ClaudeStuff
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 1. PyTorch — pick one
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU only (small download)
pip install torch                                                     # with CUDA (large download)

# 2. Everything else (numpy, scipy, pytest)
pip install -r requirements.txt
```

Training uses the GPU automatically when `torch.cuda.is_available()`. Otherwise it runs on the CPU.

## Run the tests

From the repository root:

```bash
pytest mamba_alns/tests
```

The tests check the CVRP operators (feasibility, capacity), the LP bound
(hand-checked small instances and lower-bound property), the GAT (batching,
softmax), Mamba (shape, causality), windows and N-step returns, and a short
end-to-end training run. Expect roughly 10–30 s on a CPU.

## A whole run

From the repository root:

```bash
python -m mamba_alns.train
```

This does three things:

1. **Trains** for 300 episodes with the default `Config`. Each episode prints one line:
   ```
   episode   42  eps 0.80  best <best cost found>  loss <mean training loss>
   ```
   `loss` shows `nan` for the first ~10 episodes. Learning starts only after
   `min_transitions` (1000) transitions are stored.
2. **Saves** the Q-network weights to `mamba_qnet.pt` in the current directory.
3. **Evaluates** on 20 unseen instances. On each one it runs ALNS twice: once
   with the learned greedy policy and once with uniformly random operator
   choice. It then prints:
   ```
   mean best cost  learned: <...>   random: <...>
   ```
   A lower learned cost than random means the policy is choosing better operators.

Rough CPU runtime (estimated, not measured): **15–25 minutes** for the default
run. About 70% of that is gradient steps.

### Quick run (~1 minute on CPU)

```python
from mamba_alns.train import Config, train, evaluate

cfg = Config(n_customers=20, capacity=30, iterations=50, episodes=50,
             updates_per_episode=10, min_transitions=200, eps_decay_episodes=35)
qnet = train(cfg)
evaluate(qnet, cfg, n_instances=10)
```

### Evaluate a saved model

```python
import torch
from mamba_alns.cvrp import ACTIONS
from mamba_alns.qnet import MambaQNet
from mamba_alns.train import Config, evaluate

cfg = Config()
qnet = MambaQNet(len(ACTIONS)).to(cfg.device)
qnet.load_state_dict(torch.load("mamba_qnet.pt", map_location=cfg.device))
evaluate(qnet, cfg, n_instances=20)
```

## Configuration

All training settings are fields of `Config` in `mamba_alns/train.py`. Change
them by passing keywords (`Config(episodes=500)`) or by editing the defaults.

| Field | Default | Meaning |
|---|---|---|
| `n_customers` | 50 | customers per random instance |
| `capacity` | 40 | vehicle capacity |
| `destroy_fraction` | 0.2 | share of customers removed by each destroy |
| `iterations` | 100 | ALNS iterations per episode (T) |
| `reward_eps` | 1e-3 | eps in the reward. The reward is capped at 1 / eps |
| `episodes` | 300 | training episodes (each one is a new instance) |
| `window` | 8 | K, tokens the Mamba sees |
| `n_step` | 5 | N, steps in the N-step target |
| `gamma` | 0.95 | discount |
| `lr` | 1e-4 | Adam learning rate |
| `batch_size` | 32 | transitions per gradient step |
| `buffer_episodes` | 200 | episodes kept in the replay buffer |
| `updates_per_episode` | 25 | gradient steps after each episode |
| `min_transitions` | 1000 | stored transitions before learning starts |
| `target_sync` | 250 | gradient steps between target-network copies |
| `eps_start`, `eps_end`, `eps_decay_episodes` | 1.0, 0.05, 200 | linear epsilon-greedy schedule |
| `seed` | 0 | seeds Python, numpy and torch |
| `device` | auto | `"cuda"` if available, else `"cpu"` |

Network sizes are the keyword arguments of `MambaQNet` in `mamba_alns/qnet.py`
(`gnn_hidden=64, gnn_layers=3, gnn_heads=4, d_model=64, mamba_layers=2, d_state=16`).
`train()` uses the defaults.

## Project layout

```
mamba_alns/
  cvrp.py       instances, cost, feasibility, destroy/repair operators, reward
  lp_bound.py   LP lower bound (reward reference value)
  gnn.py        solution graph construction, batching, GAT encoder
  mamba.py      Mamba block (selective scan, written as a loop)
  qnet.py       GAT -> tokens -> Mamba -> Q-values
  replay.py     episode storage, K-step windows, N-step return, replay buffer
  train.py      Config, episode loop, train step, evaluation, entry point
  tests/        pytest suite, one file per module
requirements.txt
```

## Notes and limitations

- **Episode end.** The last ALNS iteration is treated as terminal (no
  bootstrap), but the remaining budget is not part of the token, so this is
  slightly non-Markov.
- **LP bound.** The flow relaxation is weak, so expect gaps of tens of
  percent, rewards of roughly 2–5, and Q-values around reward / (1 − γ). The
  LP has about 2n² variables. It solves quickly for n = 50, but for instances
  with ~1000 customers each solve would take minutes and should be cached.
- **Mamba speed.** The selective scan is a Python loop over the window. That's
  fine for K = 8 but slow for long sequences.
- **Benchmarks.** Only random instances are generated. The CVRPLIB X set
  (Uchoa et al., 2017) is not included.
