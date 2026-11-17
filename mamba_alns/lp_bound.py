"""LP lower bound for a CVRP instance, used as the reward's reference value.

LP relaxation of the single-commodity flow (Gavish-Graves) model.
Variables, one pair per directed arc i -> j (i != j):
    x_ij in [0, 1]   arc is used
    f_ij >= 0        load still on the vehicle while it drives i -> j

    min  sum c_ij x_ij
    s.t. sum_j x_ij = 1,  sum_j x_ji = 1       each customer is left / entered once
         sum_j x_0j = sum_j x_j0               vehicles that leave the depot return
         sum_j x_0j >= ceil(total demand / Q)  enough vehicles for the total demand
         sum_j f_ji - sum_j f_ij = d_i         each customer unloads its demand
         d_j x_ij <= f_ij <= (Q - d_i) x_ij    load covers j and fits in the vehicle
         f_i0 = 0                              vehicles return empty
"""
import math

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


class _Rows:
    """Collects sparse constraint rows given as {column: coefficient}."""

    def __init__(self):
        self.rows, self.cols, self.vals, self.rhs = [], [], [], []

    def add(self, coefs, rhs):
        row = len(self.rhs)
        for col, val in coefs.items():
            self.rows.append(row)
            self.cols.append(col)
            self.vals.append(val)
        self.rhs.append(rhs)

    def matrix(self, n_cols):
        shape = (len(self.rhs), n_cols)
        A = coo_matrix((self.vals, (self.rows, self.cols)), shape=shape).tocsr()
        return A, np.array(self.rhs, dtype=float)


def lp_reference(inst):
    """Optimal value of the LP relaxation (a lower bound on the CVRP optimum)."""
    n_nodes = len(inst.demands)
    d, Q = inst.demands, inst.capacity
    arcs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
    m = len(arcs)                         # column a is x of arc a, column m + a is f of arc a
    out_arcs = [[] for _ in range(n_nodes)]
    in_arcs = [[] for _ in range(n_nodes)]
    for a, (i, j) in enumerate(arcs):
        out_arcs[i].append(a)
        in_arcs[j].append(a)

    eq, ub = _Rows(), _Rows()
    for i in range(1, n_nodes):
        eq.add({a: 1.0 for a in out_arcs[i]}, 1.0)
        eq.add({a: 1.0 for a in in_arcs[i]}, 1.0)
        flow = {m + a: 1.0 for a in in_arcs[i]}
        flow.update({m + a: -1.0 for a in out_arcs[i]})
        eq.add(flow, float(d[i]))

    depot_balance = {a: 1.0 for a in out_arcs[0]}
    depot_balance.update({a: -1.0 for a in in_arcs[0]})
    eq.add(depot_balance, 0.0)
    min_vehicles = math.ceil(d.sum() / Q)
    ub.add({a: -1.0 for a in out_arcs[0]}, -float(min_vehicles))

    for a, (i, j) in enumerate(arcs):
        ub.add({a: float(d[j]), m + a: -1.0}, 0.0)        # d_j x_ij - f_ij <= 0
        ub.add({m + a: 1.0, a: -float(Q - d[i])}, 0.0)    # f_ij - (Q - d_i) x_ij <= 0

    cost = np.concatenate([[inst.dist[i, j] for i, j in arcs], np.zeros(m)])
    bounds = [(0, 1)] * m + [(0, 0) if j == 0 else (0, None) for _, j in arcs]
    A_eq, b_eq = eq.matrix(2 * m)
    A_ub, b_ub = ub.matrix(2 * m)

    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP relaxation failed: {res.message}")
    return float(res.fun)
