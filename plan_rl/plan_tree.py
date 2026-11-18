"""Plans and the plan tree.

A plan is a sequence of (s, a, r) tuples made by rolling out the world model:
my actions come from my policy prior, the opponent's from the opponent head
and the public card from the chance head, each averaged over the belief.
Plans that share a prefix share nodes, so sampling many plans grows a tree.

Every node is a position in the public event sequence. It holds one latent
per hypothesis (possible opponent card) and the belief given the events on
the path to it.

Backup (values are per hypothesis, shape (H,)):
  my node       : q(a) = belief-weighted value of action a; my policy there is
                  softmax(q / temperature), the SAME for all hypotheses (I
                  cannot tell them apart); value = expectation under that policy
  opponent node : expectation over the opponent's actions under each hypothesis
  chance node   : expectation over the public card under each hypothesis
  leaf          : value head;  terminal: 0 (rewards sit on the edges)
"""
import torch

from .belief import LIKELIHOOD_FLOOR, normalize
from .model import MY, OPP, TERMINAL, to_event


class Node:
    def __init__(self, latents, belief, model, kind=None, legal=None):
        self.latents = latents                       # (H, d)
        self.belief = belief                         # (H,)
        pred = model.predict(latents)
        if kind is None:
            kind = int((belief @ pred.kind.softmax(-1)).argmax())
        if legal is None:
            legal = (belief @ pred.legal.sigmoid()) > 0.5
        self.kind = kind
        self.legal = legal                           # (3,) bool, used at my nodes
        self.my_prior = belief @ pred.my_policy.softmax(-1)               # (3,)
        # Outcome probabilities at opponent / chance nodes (unused at my nodes).
        outcome_logits = pred.opp_policy if kind == OPP else pred.chance
        self.outcome_probs_h = outcome_logits.softmax(-1)                  # (H, 3)
        self.value_h = pred.value                    # (H,)
        self.children = {}   # outcome x -> (child Node, reward per hypothesis (H,))
        self.q = {}          # my nodes after backup: action -> belief-weighted value
        self.policy = None   # my nodes after backup: (3,) softmax(q / temperature), 0 if not in tree

    def outcome_probs(self):
        """Opponent-action or public-card probabilities averaged over the belief (3,)."""
        return self.belief @ self.outcome_probs_h

    def child(self, x, model):
        """The node after outcome x, created with the world model on first visit."""
        if x not in self.children:
            H = len(self.belief)
            latents, reward = model.dynamics(self.latents, torch.full((H,), to_event(self.kind, x)))
            belief = self.belief
            if self.kind != MY:                      # Bayes: what the opponent / deal revealed
                belief = normalize(belief * (self.outcome_probs_h[:, x] + LIKELIHOOD_FLOOR))
            self.children[x] = (Node(latents, belief, model), reward)
        return self.children[x][0]


def sample_outcome(node, rng):
    probs = node.my_prior * node.legal if node.kind == MY else node.outcome_probs()
    probs = probs.double().numpy()
    if probs.sum() <= 0:
        return None
    return int(rng.choice(3, p=probs / probs.sum()))


def sample_plan(root, model, max_depth, rng):
    """Roll out one plan from root, adding its nodes to the tree.
    Returns [(s, a, r), ...]: the node, the outcome taken there and the predicted rewards."""
    plan, node = [], root
    while node.kind != TERMINAL and len(plan) < max_depth:
        x = sample_outcome(node, rng)
        if x is None:
            break
        child = node.child(x, model)
        plan.append((node, x, node.children[x][1]))
        node = child
    return plan


def backup(node, temperature):
    """Value of node under each hypothesis (H,); sets q and policy at my nodes."""
    if node.kind == TERMINAL:
        return torch.zeros_like(node.value_h)
    if not node.children:
        return node.value_h
    q_h = {x: reward + backup(child, temperature) for x, (child, reward) in node.children.items()}
    if node.kind == MY:
        actions = list(q_h)
        node.q = {x: float(node.belief @ q_h[x]) for x in actions}
        probs = torch.softmax(torch.tensor([node.q[x] for x in actions]) / temperature, dim=0)
        node.policy = torch.zeros(3)
        node.policy[actions] = probs
        return sum(p * q_h[x] for p, x in zip(probs, actions))
    outcomes = list(q_h)
    weights = node.outcome_probs_h[:, outcomes] + LIKELIHOOD_FLOOR       # (H, n)
    weights = weights / weights.sum(1, keepdim=True)                     # only sampled outcomes
    values = torch.stack([q_h[x] for x in outcomes], dim=1)              # (H, n)
    return (weights * values).sum(1)


@torch.no_grad()
def build_tree(model, root_latents, belief, legal, n_plans, max_depth, temperature, rng):
    """Root = my decision with the real legal actions. Expand every legal action,
    sample n_plans plans, back up. Returns (root, plans)."""
    mask = torch.zeros(3, dtype=torch.bool)
    mask[legal] = True
    root = Node(root_latents, belief, model, kind=MY, legal=mask)
    for a in legal:
        root.child(a, model)
    plans = [sample_plan(root, model, max_depth, rng) for _ in range(n_plans)]
    backup(root, temperature)
    return root, plans

