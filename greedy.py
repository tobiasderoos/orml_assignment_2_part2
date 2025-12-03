import os
from gurobipy import Model, GRB


# ---------------------------------------------------------------------
# Read the instance
# ---------------------------------------------------------------------
def read_instance(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    c = int(lines[1])
    w = list(map(int, lines[2].split()))
    p_lines = lines[3:]
    p = [list(map(int, row.split())) for row in p_lines]
    return n, c, w, p


def greedy_qkp(
    weights,
    profits,
    capacity,
    stopping_criterion=None,
):
    """
    Greedy heuristic for the 0-1 Quadratic Knapsack Problem (QKP),
    but using only linear profits for item selection.

    Items are added one‐by‐one based on their linear profit-to-weight ratio
    until the capacity is full or an optional stopping threshold is reached.

    Args:
        weights (list): List of item weights w_i.
        profits (list): List of linear profits p_i.
        capacity (int): Knapsack capacity C.
        stopping_criterion (int, optional):
            Maximum number of greedy items to select before stopping
            (e.g. stop after selecting S items).
            If None, greedy runs until no more items fit.

    Returns:
        set: A set of selected item indices.
    """
    n = len(weights)
    selected = set()
    remaining_capacity = capacity

    while remaining_capacity > 0:
        if (
            stopping_criterion is not None
            and len(selected) >= stopping_criterion
        ):
            break

        # feasible items that still fit
        candidates = [
            i
            for i in range(n)
            if i not in selected and weights[i] <= remaining_capacity
        ]

        if not candidates:
            break

        # choose item with best linear profit-to-weight ratio
        best_item = max(candidates, key=lambda i: profits[i] / weights[i])

        # select item
        selected.add(best_item)
        remaining_capacity -= weights[best_item]

    return selected


# ---------------------------------------------------------------------
# Profit computation
# ---------------------------------------------------------------------
def compute_profit(S, profits, quad_profits):
    total = sum(profits[i] for i in S)
    total += sum(quad_profits[i][j] for i in S for j in S if i < j)
    return total


def solve_ilp(weights, profits, quad_profits, capacity):
    """
    Solve the ILP for the 0-1 QPK (FULL)

    Args:
        weights: list w_i
        profits: list p_i (= diagonal of p_ij)
        quad_profits: matrix p_ij
        capacity: knapsack capacity

    Returns:
        model, x, objective_value
    """
    n = len(weights)
    m = Model("QKP_full")
    m.setParam("OutputFlag", 0)
    # Decision variables
    x = {}
    for i in range(n):
        x[i] = m.addVar(vtype=GRB.BINARY, name=f"x[{i}]")
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            y[i, j] = m.addVar(vtype=GRB.BINARY, name=f"y[{i},{j}]")
    # Capacity constraint
    m.addConstr(
        sum(weights[i] * x[i] for i in range(n)) <= capacity,
        name="capacity",
    )
    # Linking constraints: y_ij ≤ x_i, y_ij ≤ x_j
    for i in range(n):
        for j in range(i + 1, n):
            m.addConstr(y[i, j] <= x[i])
            m.addConstr(y[i, j] <= x[j])
    # Objective = linear + quadratic terms
    obj = sum(profits[i] * x[i] for i in range(n)) + sum(
        quad_profits[i][j] * y[i, j] for i in range(n) for j in range(i + 1, n)
    )
    m.setObjective(obj, GRB.MAXIMIZE)
    m.update()

    m.setParam("TimeLimit", 15)
    m.optimize()
    x_updated = {}
    for i in range(n):
        x_updated[i] = x[i].X

    obj = m.objVal if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else None
    return obj, x_updated, m.status


# ---------------------------------------------------------------------
# Reduced Integer Linear Program (ILP) for the 0-1 QKP
# ---------------------------------------------------------------------
def solve_reduced_ilp(weights, profits, quad_profits, capacity, selected):
    """
    Solve the reduced ILP for the 0-1 QPK using results from greedy.

    Args:
        weights: list w_i
        profits: list p_i
        quad_profits: matrix p_ij
        capacity: knapsack capacity
        selected: set of items chosen by greedy (S)

    Returns:
        model, x, y, objective_value
    """

    n = len(weights)
    S = set(selected)  # set of selected items of greedy solution
    R = [i for i in range(n) if i not in S]  # items not chosen by greedy

    m = Model("QKP_reduced")
    m.setParam("OutputFlag", 0)
    # Decision variables and added constraint where x_i = 1
    # TODO: fix x_i = 1 for i in S using constraints
    x = {}
    for i in R:
        x[i] = m.addVar(vtype=GRB.BINARY, name=f"x[{i}]")

    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            y[i, j] = m.addVar(vtype=GRB.BINARY, name=f"y[{i},{j}]")
            # Add constraint for pairs in S so that y_ij = 1
            if i in S and j in S:
                m.addConstr(y[i, j] == 1)

    # 4. Capacity constraint
    weight_fixed = sum(
        weights[i] for i in S
    )  # total weight of fixed items in S

    m.addConstr(
        weight_fixed + sum(weights[i] * x[i] for i in R) <= capacity,
        name="capacity",
    )

    # 5. Linking constraints: y_ij ≤ x_i, y_ij ≤ x_j
    for i in range(n):
        for j in range(i + 1, n):
            if i in S and j in S:
                continue
            elif i in S and j in R:
                m.addConstr(y[i, j] <= x[j])  # x[i] implicitly 1
            elif i in R and j in S:
                m.addConstr(y[i, j] <= x[i])  # x[j] implicitly 1
            else:  # both in R
                m.addConstr(y[i, j] <= x[i])
                m.addConstr(y[i, j] <= x[j])

    # 6. Objective = linear + quadratic terms
    constant_profits = sum(profits[i] for i in S)
    # constant profits from newly selected items
    linear_profit = sum(profits[i] * x[i] for i in R)  # Variable part
    quad_profit = sum(
        quad_profits[i][j] * y[i, j] for i in range(n) for j in range(i + 1, n)
    )
    obj = constant_profits + linear_profit + quad_profit
    m.update()
    m.setObjective(obj, GRB.MAXIMIZE)
    m.setParam("TimeLimit", 15)
    m.optimize()
    x_updated = {}
    for i in R:
        x_updated[i] = x[i].X
    obj = m.objVal if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else None
    return obj, x_updated, m.status


# ---------------------------------------------------------------------
# Run script to compare greedy, ILP, reduced ILP, and RL
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration
    # - Load RL (placeholder)
    instance_folder = "InstancesEx1/"
    instance_files = [
        f for f in os.listdir(instance_folder) if f.endswith(".txt")
    ]
    results = {
        "greedy": [],
        "ilp": [],
        "reduced_ilp": [],
        "rl": [],
    }
    for fname in instance_files:
        filepath = os.path.join(instance_folder, fname)

        # Load instance
        n, cap, weights, quad = read_instance(filepath)

        # Linear profits are the diagonal of p_ij
        profits = [quad[i][i] for i in range(n)]
        # ------------------------------------------------------
        # Run greedy
        # ------------------------------------------------------
        S_greedy = greedy_qkp(
            weights,
            profits,
            cap,
            stopping_criterion=None,
        )

        # Compute profit
        profit_greedy = compute_profit(S_greedy, profits, quad)
        results["greedy"].append((fname, len(S_greedy), profit_greedy))
        # ------------------------------------------------------
        # Run full ILP
        # ------------------------------------------------------
        ilp_obj_val, x_ilp, res_ilp = solve_ilp(weights, profits, quad, cap)
        selected_items = [i for i in range(n) if x_ilp[i] > 0.5]
        results["ilp"].append((fname, len(selected_items), ilp_obj_val))

        # ------------------------------------------------------
        # Run greedy with stopping criterion and reduced ILP
        # ------------------------------------------------------
        S_greedy_stop = greedy_qkp(weights, profits, cap, stopping_criterion=30)
        # Reduced ILP without constraints

        rilp_obj_val, x_rilp, result = solve_reduced_ilp(
            weights, profits, quad, cap, S_greedy_stop
        )
        selected_items = [i for i in x_rilp.keys() if x_rilp[i] > 0.5] + list(
            S_greedy_stop
        )

        results["reduced_ilp"].append(
            (fname, len(selected_items), rilp_obj_val)
        )
        # ------------------------------------------------------
        # Placeholder for RL results (to be filled in after training)
        # ------------------------------------------------------
    # Print results relative to full ILP
    for i in range(len(instance_files)):
        greedy_profit = results["greedy"][i][2]
        rilp_profit = results["reduced_ilp"][i][2]
        ilp_profit = results["ilp"][i][2]  # full ILP = 100%

        greedy_rel = 100 * greedy_profit / ilp_profit
        rilp_rel = 100 * rilp_profit / ilp_profit

        print(
            f"Greedy: {greedy_rel:6.2f}%   RILP: {rilp_rel:6.2f}%   Full ILP: 100.00%"
        )
