# Mamba Q-learning for ALNS operator selection (CVRP)

A GAT + Mamba Q-network that learns, with N-step Q-learning, which
(destroy, repair) pair ALNS should apply at each iteration.

| Piece | Choice |
|---|---|
| Problem | CVRP, random instances in the unit square (default 50 customers, capacity 40) |
| Action | one of 9 (destroy, repair) pairs. Destroy: random / worst / related. Repair: greedy / regret-2 / random |
| Acceptance | always accept the new solution; best_so_far tracked separately |
| Exploration | epsilon-greedy, linear decay |
| Reward | `r = 1 / ((current - ref) / ref + eps)`, ref = LP relaxation of the single-commodity flow model (scipy/HiGHS), eps = 1e-3 |
| Solution encoder | GAT over route arcs + self-loops, distance as edge feature, mean pooling |
| Sequence model | pure-PyTorch Mamba over the last K tokens |
| Token at step j | `[GAT(s_j), one_hot(a_{j-1}), r_{j-1}]` |
| Target | `Σ_{i<N} γ^i r_{t+i} + γ^N max_a Q_target(s_{t+N}, a)`; no bootstrap past the episode end |

## Layout

```
mamba_alns/
  cvrp.py     instance, cost, feasibility, destroy/repair operators, reward
  lp_bound.py LP lower bound (reward reference value)
  gnn.py      graph construction, batching, GAT encoder
  mamba.py    Mamba block (selective scan)
  qnet.py     GAT -> tokens -> Mamba -> Q-values
  replay.py   Episode storage, K-windows, N-step return, replay buffer
  train.py    Config, episode loop, train step, evaluation
  tests/
```

## Run

```
pip install -r requirements.txt
pytest mamba_alns/tests
python -m mamba_alns.train      # hyperparameters live in train.Config
```
