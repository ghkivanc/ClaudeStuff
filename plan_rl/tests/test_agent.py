import random

import numpy as np
import pytest
import torch

from plan_rl.agent import PlanningAgent
from plan_rl.cfr import uniform_policy
from plan_rl.evaluate import PolicyPlayer
from plan_rl.leduc import CALL, RAISE, InfoSet, play_hand
from plan_rl.model import OPP, WorldModel
from plan_rl.plan_tree import Node, build_tree
from plan_rl.train import Config


def make_agent(seed=0):
    torch.manual_seed(seed)
    model = WorldModel(2, d_model=16, mamba_layers=1, hidden=32)
    cfg = Config(n_types=2, n_plans=8)
    return PlanningAgent(model, 2, cfg, np.random.default_rng(seed))


def test_plays_legal_hands_against_random():
    agent = make_agent()
    rng = np.random.default_rng(0)
    py_rng = random.Random(0)
    for i in range(10):
        opponent = PolicyPlayer(uniform_policy, rng)
        players = (agent, opponent) if i % 2 == 0 else (opponent, agent)
        play_hand(*players, py_rng)          # raises on an illegal action
        assert agent.belief.sum() == pytest.approx(1.0, abs=1e-5)
        assert 1 <= agent.n_replans <= len(agent.targets)
        for target in agent.targets.values():
            assert target.sum() == pytest.approx(1.0, abs=1e-5)


def test_public_card_removes_impossible_hypotheses():
    agent = make_agent()
    agent.new_hand(seat=0, card=0)                                   # I hold a J
    first = agent.act(InfoSet(0, 0, None, ((),), [CALL, RAISE]))
    agent.act(InfoSet(0, 0, 0, ((first, CALL), ()), [CALL, RAISE]))  # public card is the other J
    card_probs = agent.belief.reshape(3, 2).sum(1)
    assert card_probs[0] == pytest.approx(0.0, abs=1e-6)


def test_follows_plan_without_replanning():
    agent = make_agent()
    agent.new_hand(seat=0, card=1)
    latents = agent.model.represent_hypotheses(0, 1, [])
    root, _ = build_tree(agent.model, latents, agent.belief, [CALL, RAISE], 8, 10,
                         np.random.default_rng(0))
    agent.node = root                                        # a plan is already in hand
    action = agent.act(InfoSet(0, 1, None, ((),), [CALL, RAISE]))
    assert action == root.choice
    assert agent.n_replans == 0


def test_surprise_drops_the_plan():
    agent = make_agent()
    agent.new_hand(seat=0, card=1)
    with torch.no_grad():
        latents = agent.model.represent_hypotheses(0, 1, [])
        node = Node(latents, agent.belief, agent.model, kind=OPP)
        expected = node.child(CALL, agent.model)             # the plan covers only CALL

    agent.cfg.surprise_threshold = 0.0
    agent.node = node
    agent._observe(OPP, CALL)
    assert agent.node is expected                            # expected reply: keep the plan

    agent.node = node
    agent._observe(OPP, RAISE)
    assert agent.node is None                                # not in the plan: surprised

    agent.cfg.surprise_threshold = 1.1
    agent.node = node
    agent._observe(OPP, CALL)
    assert agent.node is None                                # in the plan but too unlikely
