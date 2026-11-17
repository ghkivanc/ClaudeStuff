import numpy as np
import pytest

from mamba_alns.cvrp import (ACTIONS, Instance, apply_action, initial_solution, random_instance,
                             solution_cost)
from mamba_alns.lp_bound import lp_reference


def make_instance(coords, demands, capacity):
    coords = np.array(coords, dtype=float)
    dist = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(-1))
    return Instance(coords, np.array(demands), capacity, dist)


def test_single_customer_is_exact():
    # Only possible tour: depot -> 1 -> depot, so the LP is tight.
    inst = make_instance([[0, 0], [3, 4]], [0, 2], 10)
    assert lp_reference(inst) == pytest.approx(10.0)


def test_fleet_size_constraint():
    # Demands 6 + 6 > Q = 10, so at least 2 vehicles: both customers get their own route.
    inst = make_instance([[0, 0], [1, 0], [0, 2]], [0, 6, 6], 10)
    assert lp_reference(inst) == pytest.approx(2 * 1 + 2 * 2)


def test_capacity_forces_more_than_tsp():
    # Without capacity, one tour of the square costs 4. With Q = 1 each customer
    # needs its own trip, which the flow constraints must enforce.
    coords = [[0, 0], [1, 0], [1, 1], [0, 1]]
    tight = make_instance(coords, [0, 1, 1, 1], 1)
    loose = make_instance(coords, [0, 1, 1, 1], 10)
    assert lp_reference(loose) == pytest.approx(4.0)
    assert lp_reference(tight) == pytest.approx(2 * (1 + np.sqrt(2) + 1))


def test_lp_is_lower_bound_on_alns_solutions():
    rng = np.random.default_rng(0)
    inst = random_instance(15, 20, rng)
    ref = lp_reference(inst)
    routes = initial_solution(inst)
    assert 0 < ref <= solution_cost(routes, inst) + 1e-6
    for action in range(len(ACTIONS)):
        routes = apply_action(action, routes, inst, 4, rng)
        assert ref <= solution_cost(routes, inst) + 1e-6
