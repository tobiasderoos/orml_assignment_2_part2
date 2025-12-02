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


# ---------------------------------------------------------------------
# Greedy heuristic
# ---------------------------------------------------------------------
def greedy_qkp(
    weights,
    profits,
    quad_profits,
    capacity,
    stopping_criterion=None,
):
    """
    Greedy heuristic for the 0-1 Quadratic Knapsack Problem.

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
            if len(selected) == stopping_criterion:
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
    m.setParam("TimeLimit", 15)
    m.optimize()
    return m, x, m.objVal


# ---------------------------------------------------------------------
# Reduced Integer Linear Program (ILP) for the 0-1 QKP
# ---------------------------------------------------------------------
def solve_reduced_ilp(weights, profits, quad_profits, capacity, selected):
    """
    Solve the reduced ILP for the 0-1 QPK.

    Args:
        weights: list w_i
        profits: list p_i (= diagonal of p_ij)
        quad_profits: matrix p_ij
        capacity: knapsack capacity
        selected: set of items chosen by greedy (S)

    Returns:
        model, x, y, objective_value
    """

    n = len(weights)
    S = set(selected)
    R = [i for i in range(n) if i not in S]  # set excluding S

    m = Model("QKP_reduced")

    # Decision variables and added constraint where x_i = 1
    x = {}
    for i in range(n):
        x[i] = m.addVar(vtype=GRB.BINARY, name=f"x[{i}]")
        if i in S:
            # fixed variable
            m.addConstr(x[i] == 1)

    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            y[i, j] = m.addVar(vtype=GRB.BINARY, name=f"y[{i},{j}]")
            # Add constraint for pairs in S
            if i in S and j in S:
                m.addConstr(y[i, j] == 1)

    # 4. Capacity constraint
    weight_fixed = sum(weights[i] for i in S)

    m.addConstr(
        weight_fixed + sum(weights[i] * x[i] for i in R) <= capacity,
        name="capacity",
    )

    # 5. Linking constraints: y_ij ≤ x_i, y_ij ≤ x_j
    for i in range(n):
        for j in range(i + 1, n):
            if i in S and j in S:
                continue
            m.addConstr(y[i, j] <= x[i])
            m.addConstr(y[i, j] <= x[j])

    # 6. Objective = linear + quadratic terms
    # Add constant part from pairs inside S (already selected items)
    constant_profit = sum(profits[i] for i in S) + sum(
        quad_profits[i][j] for i in S for j in S if i < j
    )

    obj = (
        constant_profit
        + sum(profits[i] * x[i] for i in R)
        + sum(quad_profits[i][j] * y[i, j] for i in R for j in R if i < j)
        + sum(quad_profits[i][j] * x[j] for i in S for j in R)
    )

    m.setObjective(obj, GRB.MAXIMIZE)
    m.setParam("TimeLimit", 15)

    m.optimize()

    return m, x, m.objVal


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration
    # - Load RL (placeholder)
    instance_folder = "InstancesEx1/"
    instance_files = [
        f for f in os.listdir(instance_folder) if f.endswith(".txt")
    ]
    results = {"greedy": [], "ilp": [], "reduced_ilp": [], "rl": []}
    for fname in instance_files:
        filepath = os.path.join(instance_folder, fname)

        # Load instance
        n, cap, weights, quad = read_instance(filepath)

        # Linear profits are the diagonal of p_ij
        profits = [quad[i][i] for i in range(n)]
        # ------------------------------------------------------
        # Run greedy
        # ------------------------------------------------------
        S = greedy_qkp(
            weights,
            profits,
            quad,
            cap,
            stopping_criterion=None,
        )

        # Compute profit
        profit_greedy = compute_profit(S, profits, quad)
        results["greedy"].append((fname, len(S), profit_greedy))
        # ------------------------------------------------------
        # Run full ILP
        # ------------------------------------------------------
        m, x, objective_value = solve_ilp(weights, profits, quad, cap)
        selected_items = [i for i in range(n) if x[i].x > 0.5]
        results["ilp"].append((fname, len(selected_items), objective_value))

        # ------------------------------------------------------
        # Run greedy with stopping criterion adn reduced ILP
        # ------------------------------------------------------
        S = greedy_qkp(weights, profits, quad, cap, stopping_criterion=5)
        # Reduced ILP
        m, x, objective_value = solve_reduced_ilp(
            weights, profits, quad, cap, S
        )
        selected_items = [i for i in range(n) if x[i].x > 0.5]
        results["reduced_ilp"].append(
            (fname, len(selected_items), objective_value)
        )
        # ------------------------------------------------------
        # Placeholder for RL results (to be filled in after training)
        # ------------------------------------------------------

    # Print results
    for i in range(len(instance_files)):
        fname = instance_files[i]
        greedy_len = results["greedy"][i][1]
        greedy_profit = results["greedy"][i][2]
        ilp_len = results["ilp"][i][1]
        ilp_profit = results["ilp"][i][2]
        print(
            f"Instance: {fname} | Greedy: items={greedy_len}, profit={greedy_profit} | ILP: items={ilp_len}, profit={ilp_profit}, diff={ilp_profit - greedy_profit}"
        )
