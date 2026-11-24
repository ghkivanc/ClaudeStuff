"""Map every ReLU MLP to one fixed representative of its weight-symmetry class.

Two symmetries leave a ReLU MLP's function unchanged:
  * scale: multiply a hidden unit's incoming row and bias by c > 0 and divide
    its outgoing column by c;
  * permutation: reorder the hidden units of a layer.

Two passes, so the sort key of a layer never depends on a scale that is
removed later:
  1. Remove scale in every hidden layer, input to output. Each unit's incoming
     row [W_in[j, :], b_j] gets unit norm.
  2. Sort the units of every hidden layer by ||W_out[:, j]|| descending, with
     b_j (descending) breaking ties.
The output layer keeps its order, because outputs map to fixed actions.
"""

import torch


def canonicalize(spec, theta, eps=1e-12):
    """theta (..., P) -> canonical theta (..., P), computing the same function."""
    layers = [[W, b] for W, b in spec.unflatten(theta)]
    num_hidden = len(layers) - 1

    # Pass 1: remove scale. ReLU(c x) = c ReLU(x) for c > 0.
    for l in range(num_hidden):
        W, b = layers[l]
        norm = torch.cat([W, b.unsqueeze(-1)], dim=-1).norm(dim=-1).clamp_min(eps)
        layers[l][0] = W / norm.unsqueeze(-1)
        layers[l][1] = b / norm
        layers[l + 1][0] = layers[l + 1][0] * norm.unsqueeze(-2)

    # Pass 2: remove permutation.
    for l in range(num_hidden):
        W, b = layers[l]
        W_out = layers[l + 1][0]
        key = W_out.norm(dim=-2)

        # Sort by b first, then by key with a stable sort, so b breaks key ties.
        by_bias = torch.argsort(b, dim=-1, descending=True, stable=True)
        by_key = torch.argsort(key.gather(-1, by_bias), dim=-1, descending=True, stable=True)
        order = by_bias.gather(-1, by_key)

        layers[l][0] = W.gather(-2, order.unsqueeze(-1).expand_as(W))
        layers[l][1] = b.gather(-1, order)
        layers[l + 1][0] = W_out.gather(-1, order.unsqueeze(-2).expand_as(W_out))

    return spec.flatten(layers)
