# ---------------------------------------------------------------------
# Greedy heuristic
# ---------------------------------------------------------------------
def greedy_qkp_quadratic(
    weights,
    profits,
    quad_profits,
    capacity,
    stopping_criterion=None,
):
    """
    Greedy heuristic for the QPS.

    Args:
        weights: list of w_i
        profits: list of p_i
        quad_profits: matrix p_ij (size n x n)
        capacity: knapsack capacity c
        stopping_criterion: function S -> bool (optional)

    Returns:
        selected: set of chosen items
    """
    n = len(weights)
    selected = set()
    remaining_capacity = capacity
    while remaining_capacity > 0:
        if stopping_criterion is not None:
            if len(selected) >= stopping_criterion:
                break

        # All feasible items that are less than remaining capacity
        candidates = [
            i
            for i in range(n)
            if i not in selected and weights[i] <= remaining_capacity
        ]

        if not candidates:
            break

        # Compute marginal profit ratio for each candidate
        best_item = None
        best_ratio = -1

        for i in candidates:
            # Marginal profit: linear + interactions with already selected items / weight
            profits_i = profits[i]
            int_profits_ij = sum(quad_profits[i][j] for j in selected)
            marginal_profit = profits_i + int_profits_ij
            ratio = marginal_profit / weights[i]

            if ratio > best_ratio:
                best_ratio = ratio
                best_item = i

        # Add the item to the knapsack
        selected.add(best_item)
        remaining_capacity -= weights[best_item]

    return selected


def solve_reduced_ilp_constraints(
    weights, profits, quad_profits, capacity, selected
):
    """
    Solve the reduced ILP for the 0-1 QKP using results from greedy.

    This methods uses all items as decision variables, but constraining
    then to be equal to 1 for those in selected set S.
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
    # R = [i for i in range(n) if i not in S]  # items not chosen by greedy

    m = Model("QKP_reduced_constraints")

    # Decision variables - create for ALL items
    x = {}
    for i in range(n):
        x[i] = m.addVar(vtype=GRB.BINARY, name=f"x[{i}]")
        # Fix variables for items in S using explicit constraints
        if i in S:
            m.addConstr(x[i] == 1, name=f"fix_x{i}")

    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            y[i, j] = m.addVar(vtype=GRB.BINARY, name=f"y[{i},{j}]")
            if i in S and j in S:
                m.addConstr(y[i, j] == 1)

    # Capacity constraint
    m.addConstr(
        sum(weights[i] * x[i] for i in range(n)) <= capacity,
        name="capacity",
    )

    # Linking constraints: y_ij ≤ x_i, y_ij ≤ x_j
    for i in range(n):
        for j in range(i + 1, n):
            m.addConstr(y[i, j] <= x[i], name=f"link_{i}_{j}_1")
            m.addConstr(y[i, j] <= x[j], name=f"link_{i}_{j}_2")

    # Objective = linear + quadratic terms
    obj = sum(profits[i] * x[i] for i in range(n)) + sum(
        quad_profits[i][j] * y[i, j] for i in range(n) for j in range(i + 1, n)
    )
    m.update()
    m.setObjective(obj, GRB.MAXIMIZE)
    m.setParam("TimeLimit", 15)
    m.optimize()

    x_updated = {}
    for i in range(n):
        x_updated[i] = x[i].X

    obj = m.objVal if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else None
    return obj, x_updated, m.status
