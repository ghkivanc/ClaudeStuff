import numpy as np
import torch

from mamba_alns.cvrp import initial_solution, random_instance
from mamba_alns.gnn import (GATEncoder, batch_graphs, node_features, scatter_softmax,
                            solution_edges)


def make_graph(seed, n=10):
    inst = random_instance(n, 20, np.random.default_rng(seed))
    routes = initial_solution(inst)
    feats = node_features(inst)
    edges = solution_edges(routes, n + 1)
    dist = torch.as_tensor(inst.dist, dtype=torch.float32)
    return feats, edges, dist


def test_solution_edges_count():
    edges = solution_edges([[1, 2], [3]], n_nodes=4)
    # route arcs: 0-1, 1-2, 2-0, 0-3, 3-0 = 5 arcs, both directions = 10, + 4 self-loops
    assert edges.shape == (2, 14)


def test_node_features():
    inst = random_instance(5, 20, np.random.default_rng(0))
    feats = node_features(inst)
    assert feats.shape == (6, 4)
    assert feats[0, 3] == 1 and feats[1:, 3].sum() == 0


def test_scatter_softmax_sums_to_one_per_node():
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    scores = torch.randn(6, 3)
    alpha = scatter_softmax(scores, index, 3)
    sums = torch.zeros(3, 3).index_add_(0, index, alpha)
    assert torch.allclose(sums, torch.ones(3, 3), atol=1e-6)


def test_encoder_output_shape():
    graphs = [make_graph(s) for s in range(3)]
    enc = GATEncoder(hidden=32, n_layers=2, heads=4)
    out = enc(*batch_graphs(graphs))
    assert out.shape == (3, 32)


def test_batching_does_not_mix_graphs():
    torch.manual_seed(0)
    enc = GATEncoder(hidden=32, n_layers=2, heads=4)
    g0, g1 = make_graph(0), make_graph(1, n=7)
    alone = enc(*batch_graphs([g1]))
    batched = enc(*batch_graphs([g0, g1]))
    assert torch.allclose(alone[0], batched[1], atol=1e-5)
