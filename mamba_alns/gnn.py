"""Graph attention encoder for a CVRP solution (pure PyTorch, no PyG).

Graph of one solution:
  nodes : depot + customers, features [x, y, demand / capacity, is_depot]
  edges : the arcs of the solution's routes (both directions) + self-loops
  edge feature : euclidean distance of the arc
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NODE_DIM = 4


def node_features(inst):
    n_nodes = len(inst.demands)
    feats = torch.zeros(n_nodes, NODE_DIM)
    feats[:, 0:2] = torch.as_tensor(inst.coords, dtype=torch.float32)
    feats[:, 2] = torch.as_tensor(inst.demands / inst.capacity, dtype=torch.float32)
    feats[0, 3] = 1.0
    return feats


def solution_edges(routes, n_nodes):
    """(2, E) tensor of [source, target] node indices."""
    src, dst = [], []
    for r in routes:
        path = [0] + r + [0]
        for a, b in zip(path[:-1], path[1:]):
            src += [a, b]
            dst += [b, a]
    # Self-loops so every node (including unvisited-looking ones) attends to itself.
    src += list(range(n_nodes))
    dst += list(range(n_nodes))
    return torch.tensor([src, dst], dtype=torch.long)


def batch_graphs(graphs):
    """Merge graphs into one big disconnected graph.

    graphs: list of (node_feats, edge_index, dist) with dist a (n, n) tensor.
    Returns x, edge_index, edge_attr, graph_id (node -> graph), n_graphs.
    """
    xs, edges, attrs, ids = [], [], [], []
    offset = 0
    for g, (feats, edge_index, dist) in enumerate(graphs):
        n = feats.shape[0]
        xs.append(feats)
        edges.append(edge_index + offset)
        attrs.append(dist[edge_index[0], edge_index[1]].unsqueeze(-1))
        ids.append(torch.full((n,), g, dtype=torch.long))
        offset += n
    return (torch.cat(xs), torch.cat(edges, dim=1), torch.cat(attrs),
            torch.cat(ids), len(graphs))


def scatter_softmax(scores, index, n_nodes):
    """Softmax of scores (E, H) over all edges that share the same target node."""
    idx = index.unsqueeze(-1).expand_as(scores)
    max_per_node = torch.full((n_nodes, scores.shape[1]), float("-inf"), device=scores.device)
    max_per_node = max_per_node.scatter_reduce(0, idx, scores, reduce="amax")
    exp = (scores - max_per_node[index]).exp()
    denom = torch.zeros(n_nodes, scores.shape[1], device=scores.device).index_add_(0, index, exp)
    return exp / denom[index]


class GATLayer(nn.Module):
    """Multi-head graph attention where the edge distance enters both the
    attention score and the message."""

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.head_dim = heads, dim // heads
        self.W = nn.Linear(dim, dim, bias=False)
        self.W_edge = nn.Linear(1, dim, bias=False)
        self.att = nn.Parameter(torch.randn(heads, 3 * self.head_dim) * 0.1)

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index
        n, e = h.shape[0], edge_index.shape[1]
        Wh = self.W(h).view(n, self.heads, self.head_dim)
        We = self.W_edge(edge_attr).view(e, self.heads, self.head_dim)

        pair = torch.cat([Wh[dst], Wh[src], We], dim=-1)            # (E, H, 3*Dh)
        scores = F.leaky_relu((pair * self.att).sum(-1), 0.2)        # (E, H)
        alpha = scatter_softmax(scores, dst, n)                      # (E, H)

        messages = alpha.unsqueeze(-1) * (Wh[src] + We)              # (E, H, Dh)
        out = torch.zeros_like(Wh).index_add_(0, dst, messages)
        return out.reshape(n, -1)


class GATEncoder(nn.Module):
    """Stack of GAT layers (residual + LayerNorm), mean-pooled to one vector per graph."""

    def __init__(self, hidden, n_layers, heads):
        super().__init__()
        self.hidden = hidden
        self.input = nn.Linear(NODE_DIM, hidden)
        self.layers = nn.ModuleList([GATLayer(hidden, heads) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

    def forward(self, x, edge_index, edge_attr, graph_id, n_graphs):
        h = self.input(x)
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + F.elu(layer(h, edge_index, edge_attr)))
        summed = torch.zeros(n_graphs, self.hidden, device=h.device).index_add_(0, graph_id, h)
        counts = torch.bincount(graph_id, minlength=n_graphs).clamp(min=1).unsqueeze(-1)
        return summed / counts
