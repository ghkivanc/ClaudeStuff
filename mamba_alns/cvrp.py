"""CVRP instance, solution helpers and ALNS destroy/repair operators.

A solution is a list of routes. Each route is a list of customer indices
(1..n). The depot (index 0) is implicit at the start and end of each route.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Instance:
    coords: np.ndarray   # (n+1, 2), row 0 is the depot
    demands: np.ndarray  # (n+1,), demands[0] == 0
    capacity: int
    dist: np.ndarray     # (n+1, n+1) euclidean distances

    @property
    def n_customers(self):
        return len(self.demands) - 1


def random_instance(n_customers, capacity, rng):
    """Uniform coordinates in the unit square, integer demands in [1, 9]."""
    coords = rng.random((n_customers + 1, 2))
    demands = rng.integers(1, 10, size=n_customers + 1)
    demands[0] = 0
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    return Instance(coords, demands, capacity, dist)


# ---------------------------------------------------------------- solutions

def route_cost(route, inst):
    if not route:
        return 0.0
    path = [0] + route + [0]
    return float(sum(inst.dist[a, b] for a, b in zip(path[:-1], path[1:])))


def solution_cost(routes, inst):
    return sum(route_cost(r, inst) for r in routes)


def route_load(route, inst):
    return int(sum(inst.demands[c] for c in route))


def is_feasible(routes, inst):
    """Every customer visited exactly once and no route over capacity."""
    visited = sorted(c for r in routes for c in r)
    if visited != list(range(1, inst.n_customers + 1)):
        return False
    return all(route_load(r, inst) <= inst.capacity for r in routes)


def initial_solution(inst):
    """Nearest-neighbour construction: extend the current route with the
    closest customer that still fits, otherwise start a new route."""
    unvisited = set(range(1, inst.n_customers + 1))
    routes = []
    while unvisited:
        route, load, last = [], 0, 0
        while True:
            fits = [c for c in unvisited if load + inst.demands[c] <= inst.capacity]
            if not fits:
                break
            nxt = min(fits, key=lambda c: inst.dist[last, c])
            route.append(nxt)
            load += inst.demands[nxt]
            unvisited.remove(nxt)
            last = nxt
        routes.append(route)
    return routes


def copy_routes(routes):
    return [list(r) for r in routes]


# ---------------------------------------------------------------- destroy
# Each destroy operator: (routes, inst, k, rng) -> (partial_routes, removed)

def _remove(routes, to_remove):
    to_remove = set(to_remove)
    partial = [[c for c in r if c not in to_remove] for r in routes]
    return [r for r in partial if r]


def random_removal(routes, inst, k, rng):
    customers = [c for r in routes for c in r]
    removed = [int(c) for c in rng.choice(customers, size=k, replace=False)]
    return _remove(routes, removed), removed


def worst_removal(routes, inst, k, rng):
    """Remove the k customers whose removal saves the most distance."""
    savings = []
    for r in routes:
        path = [0] + r + [0]
        for i in range(1, len(path) - 1):
            prev, c, nxt = path[i - 1], path[i], path[i + 1]
            saving = inst.dist[prev, c] + inst.dist[c, nxt] - inst.dist[prev, nxt]
            savings.append((saving, c))
    savings.sort(reverse=True)
    removed = [c for _, c in savings[:k]]
    return _remove(routes, removed), removed


def related_removal(routes, inst, k, rng):
    """Pick a random seed customer and remove it plus its k-1 nearest customers."""
    seed = int(rng.integers(1, inst.n_customers + 1))
    order = np.argsort(inst.dist[seed])
    removed = [int(c) for c in order if c != 0][:k]
    return _remove(routes, removed), removed


# ---------------------------------------------------------------- repair
# Each repair operator: (partial_routes, removed, inst, rng) -> routes

NEW_ROUTE = -1


def insertion_options(c, routes, loads, inst):
    """All feasible (cost, route_idx, position) insertions of customer c,
    sorted by cost. route_idx == NEW_ROUTE means opening a new route."""
    options = [(2 * inst.dist[0, c], NEW_ROUTE, 0)]
    for ri, r in enumerate(routes):
        if loads[ri] + inst.demands[c] > inst.capacity:
            continue
        path = [0] + r + [0]
        for pos in range(len(r) + 1):
            prev, nxt = path[pos], path[pos + 1]
            cost = inst.dist[prev, c] + inst.dist[c, nxt] - inst.dist[prev, nxt]
            options.append((cost, ri, pos))
    options.sort(key=lambda o: o[0])
    return options


def _insert(c, route_idx, pos, routes, loads, inst):
    c = int(c)  # rng.permutation yields numpy ints
    if route_idx == NEW_ROUTE:
        routes.append([c])
        loads.append(int(inst.demands[c]))
    else:
        routes[route_idx].insert(pos, c)
        loads[route_idx] += int(inst.demands[c])


def greedy_insertion(routes, removed, inst, rng):
    """Insert customers (in random order) at their cheapest feasible position."""
    routes = copy_routes(routes)
    loads = [route_load(r, inst) for r in routes]
    for c in rng.permutation(removed):
        _, ri, pos = insertion_options(c, routes, loads, inst)[0]
        _insert(c, ri, pos, routes, loads, inst)
    return routes


def regret_insertion(routes, removed, inst, rng):
    """Regret-2: repeatedly insert the customer with the largest difference
    between its best and second-best insertion cost."""
    routes = copy_routes(routes)
    loads = [route_load(r, inst) for r in routes]
    remaining = list(removed)
    while remaining:
        best_c, best_regret, best_option = None, -1.0, None
        for c in remaining:
            options = insertion_options(c, routes, loads, inst)
            regret = options[1][0] - options[0][0] if len(options) > 1 else 0.0
            if regret > best_regret:
                best_c, best_regret, best_option = c, regret, options[0]
        _, ri, pos = best_option
        _insert(best_c, ri, pos, routes, loads, inst)
        remaining.remove(best_c)
    return routes


def random_insertion(routes, removed, inst, rng):
    """Insert customers (in random order) at a uniformly random feasible position."""
    routes = copy_routes(routes)
    loads = [route_load(r, inst) for r in routes]
    for c in rng.permutation(removed):
        options = insertion_options(c, routes, loads, inst)
        _, ri, pos = options[rng.integers(len(options))]
        _insert(c, ri, pos, routes, loads, inst)
    return routes


DESTROY_OPS = [random_removal, worst_removal, related_removal]
REPAIR_OPS = [greedy_insertion, regret_insertion, random_insertion]

# One action = one (destroy, repair) pair.
ACTIONS = [(d, r) for d in DESTROY_OPS for r in REPAIR_OPS]


def apply_action(action, routes, inst, k, rng):
    destroy, repair = ACTIONS[action]
    partial, removed = destroy(routes, inst, k, rng)
    return repair(partial, removed, inst, rng)


# ---------------------------------------------------------------- reward

def reward(current_cost, best_cost):
    """e^(best - current). best_cost must already include the current
    solution, so the reward is in (0, 1] and equals 1 at a new best."""
    return float(np.exp(best_cost - current_cost))
