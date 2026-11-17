import math

import numpy as np
import pytest

from mamba_alns.cvrp import (ACTIONS, DESTROY_OPS, Instance, apply_action, initial_solution,
                             is_feasible, random_instance, reward, route_load, solution_cost)


@pytest.fixture
def inst():
    return random_instance(20, 30, np.random.default_rng(0))


def test_instance_shapes(inst):
    assert inst.coords.shape == (21, 2)
    assert inst.demands[0] == 0
    assert inst.dist.shape == (21, 21)
    assert np.allclose(inst.dist, inst.dist.T)


def test_cost_on_hand_built_instance():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    dist = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(-1))
    tiny = Instance(coords, np.array([0, 1, 1]), 10, dist)
    # depot -> 1 -> 2 -> depot = 1 + 1 + sqrt(2)
    assert solution_cost([[1, 2]], tiny) == pytest.approx(2 + math.sqrt(2))


def test_initial_solution_feasible(inst):
    assert is_feasible(initial_solution(inst), inst)


@pytest.mark.parametrize("destroy", DESTROY_OPS)
def test_destroy_removes_exactly_k(inst, destroy):
    routes = initial_solution(inst)
    partial, removed = destroy(routes, inst, 5, np.random.default_rng(1))
    remaining = [c for r in partial for c in r]
    assert len(removed) == 5
    assert len(set(removed)) == 5
    assert sorted(remaining + list(removed)) == list(range(1, 21))
    assert all(len(r) > 0 for r in partial)


@pytest.mark.parametrize("action", range(len(ACTIONS)))
def test_every_action_gives_feasible_solution(inst, action):
    rng = np.random.default_rng(2)
    routes = initial_solution(inst)
    for _ in range(5):
        routes = apply_action(action, routes, inst, 6, rng)
        assert is_feasible(routes, inst)


def test_tight_capacity_respected():
    inst = random_instance(15, 10, np.random.default_rng(3))
    routes = initial_solution(inst)
    rng = np.random.default_rng(4)
    for action in range(len(ACTIONS)):
        routes = apply_action(action, routes, inst, 5, rng)
        assert all(route_load(r, inst) <= inst.capacity for r in routes)


def test_reward():
    assert reward(5.0, 5.0) == pytest.approx(1.0)
    assert reward(6.0, 5.0) == pytest.approx(math.exp(-1.0 / 5.0))
    assert 0.0 < reward(100.0, 5.0) <= 1.0
    # scale invariant: same relative gap, same reward
    assert reward(6000.0, 5000.0) == pytest.approx(reward(6.0, 5.0))
