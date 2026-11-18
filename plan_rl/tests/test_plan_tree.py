import numpy as np
import pytest
import torch

from plan_rl.belief import initial_belief
from plan_rl.leduc import CALL, RAISE
from plan_rl.model import MY, OPP, WorldModel
from plan_rl.plan_tree import Node, backup, build_tree, policy_target


def setup():
    torch.manual_seed(0)
    model = WorldModel(2, d_model=16, mamba_layers=1, hidden=32)
    latents = model.represent_hypotheses(0, 0, [])
    belief = initial_belief(0, n_types=2, n_known_types=2)
    return model, latents, belief


def count_nodes(node):
    return 1 + sum(count_nodes(child) for child, _ in node.children.values())


def test_root_expands_legal_actions_and_chooses_one():
    model, latents, belief = setup()
    root, plans = build_tree(model, latents, belief, [CALL, RAISE], 16, 10,
                             np.random.default_rng(0))
    assert root.kind == MY
    assert set(root.children) == {CALL, RAISE}
    assert root.choice in (CALL, RAISE)
    assert root.choice == max(root.q, key=root.q.get)


def test_plans_share_prefixes():
    model, latents, belief = setup()
    root, plans = build_tree(model, latents, belief, [CALL, RAISE], 16, 10,
                             np.random.default_rng(0))
    assert len(plans) == 16
    assert all(plan[0][0] is root for plan in plans if plan)
    assert count_nodes(root) - 1 < sum(len(plan) for plan in plans)


def test_my_value_is_belief_weighted():
    model, latents, belief = setup()
    root, _ = build_tree(model, latents, belief, [CALL, RAISE], 8, 10, np.random.default_rng(1))
    for a, (child, reward) in root.children.items():
        assert root.q[a] == pytest.approx(float(belief @ (reward + backup(child))), abs=1e-5)


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


def test_policy_target():
    model, latents, belief = setup()
    root, _ = build_tree(model, latents, belief, [CALL, RAISE], 8, 10, np.random.default_rng(2))
    target = policy_target(root, temperature=1.0)
    assert target.sum() == pytest.approx(1.0)
    assert target[0] == 0                                    # fold not legal at the root
