import numpy as np
import pytest
import torch

from plan_rl.belief import initial_belief
from plan_rl.leduc import CALL, FOLD, RAISE
from plan_rl.model import MY, OPP, WorldModel
from plan_rl.plan_tree import Node, backup, build_tree

T = 1.0   # temperature


def setup():
    torch.manual_seed(0)
    model = WorldModel(d_model=16, mamba_layers=1, hidden=32)
    latents = model.represent_hypotheses(0, 0, [])
    return model, latents, initial_belief(0)


def tree(n_plans=16, seed=0):
    model, latents, belief = setup()
    root, plans = build_tree(model, latents, belief, [CALL, RAISE], n_plans, 10, T,
                             np.random.default_rng(seed))
    return root, plans, belief


def count_nodes(node):
    return 1 + sum(count_nodes(child) for child, _ in node.children.values())


def test_root_expands_legal_actions():
    root, _, _ = tree()
    assert root.kind == MY
    assert set(root.children) == {CALL, RAISE}


def test_root_policy_is_softmax_of_values():
    root, _, _ = tree()
    q = torch.tensor([root.q[CALL], root.q[RAISE]])
    expected = torch.softmax(q / T, dim=0)
    assert root.policy[FOLD] == 0                            # fold not legal at the root
    assert root.policy[[CALL, RAISE]].tolist() == pytest.approx(expected.tolist())


def test_plans_share_prefixes():
    root, plans, _ = tree()
    assert len(plans) == 16
    assert all(plan[0][0] is root for plan in plans if plan)
    assert count_nodes(root) - 1 < sum(len(plan) for plan in plans)


def test_my_q_is_belief_weighted():
    root, _, belief = tree(n_plans=8, seed=1)
    for a, (child, reward) in root.children.items():
        assert root.q[a] == pytest.approx(float(belief @ (reward + backup(child, T))), abs=1e-5)


def test_my_value_is_expectation_under_policy():
    root, _, belief = tree(n_plans=8, seed=2)
    value = backup(root, T)
    expected = sum(root.policy[a] * root.q[a] for a in root.children)
    assert float(belief @ value) == pytest.approx(float(expected), abs=1e-5)


def test_opponent_child_belief_is_bayes():
    model, latents, belief = setup()
    with torch.no_grad():
        node = Node(latents, belief, model, kind=OPP)
        child = node.child(RAISE, model)
    expected = belief * node.outcome_probs_h[:, RAISE]
    assert torch.allclose(child.belief, expected / expected.sum(), atol=1e-5)


def test_my_child_keeps_belief():
    model, latents, belief = setup()
    with torch.no_grad():
        node = Node(latents, belief, model, kind=MY)
        child = node.child(CALL, model)
    assert torch.equal(child.belief, belief)
