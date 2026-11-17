"""Q-network: GAT encodes each solution in the window, Mamba reads the window.

Token at step j = [GNN(s_j), one_hot(a_{j-1}), r_{j-1}]
Q(window ending at t) = Linear(Mamba(tokens)[last token])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import GATEncoder, batch_graphs
from .mamba import ResidualMamba


class MambaQNet(nn.Module):
    def __init__(self, n_actions, gnn_hidden=64, gnn_layers=3, gnn_heads=4,
                 d_model=64, mamba_layers=2, d_state=16):
        super().__init__()
        self.n_actions = n_actions
        self.gnn = GATEncoder(gnn_hidden, gnn_layers, gnn_heads)
        self.token_proj = nn.Linear(gnn_hidden + n_actions + 1, d_model)
        self.blocks = nn.ModuleList(
            [ResidualMamba(d_model, d_state=d_state) for _ in range(mamba_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_actions)

    def forward(self, graphs, prev_actions, prev_rewards):
        """
        graphs       : list of B*K graphs (window-major: window 0's K graphs first)
        prev_actions : (B, K) long, -1 means "no previous action"
        prev_rewards : (B, K) float
        returns      : (B, n_actions) Q-values for the last step of each window
        """
        device = self.head.weight.device
        batch, k = prev_actions.shape

        x, edge_index, edge_attr, graph_id, n_graphs = batch_graphs(graphs)
        emb = self.gnn(x.to(device), edge_index.to(device), edge_attr.to(device),
                       graph_id.to(device), n_graphs)
        emb = emb.view(batch, k, -1)

        prev_actions = prev_actions.to(device)
        action_onehot = F.one_hot(prev_actions.clamp(min=0), self.n_actions).float()
        action_onehot = action_onehot * (prev_actions >= 0).unsqueeze(-1)  # zero for "none"
        rewards = prev_rewards.to(device).unsqueeze(-1)

        tokens = self.token_proj(torch.cat([emb, action_onehot, rewards], dim=-1))
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens[:, -1]))
