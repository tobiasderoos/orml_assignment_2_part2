import os
from gurobipy import Model, GRB


# ==============================================================
# 1. Read instance
# ==============================================================
def read_instance(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    c = int(lines[1])
    w = list(map(int, lines[2].split()))
    p_lines = lines[3:]
    p = [list(map(int, row.split())) for row in p_lines]
    return n, c, w, p


# ==============================================================
# 2. Greedy heuristic (simplified + fixed stopping)
# ==============================================================
def greedy_qkp(weights, profits, quad_profits, capacity, max_items=None):
    n = len(weights)
    selected = set()
    cap_left = capacity

    while cap_left > 0:
        if max_items is not None and len(selected) >= max_items:
            break

        candidates = [
            i for i in range(n) if i not in selected and weights[i] <= cap_left
        ]

        if not candidates:
            break

        best_item = None
        best_ratio = -1

        for i in candidates:
            marginal = profits[i] + sum(quad_profits[i][j] for j in selected)
            ratio = marginal / weights[i]
            if ratio > best_ratio:
                best_ratio = ratio
                best_item = i

        selected.add(best_item)
        cap_left -= weights[best_item]

    return selected


# ==============================================================
# 3. Compute profit
# ==============================================================
def compute_profit(S, profits, quad_profits):
    total = sum(profits[i] for i in S)
    total += sum(quad_profits[i][j] for i in S for j in S if i < j)
    return total


# ==============================================================
# 4. Full ILP
# ==============================================================
def solve_ilp(weights, profits, quad_profits, capacity):
    n = len(weights)
    m = Model("QKP_full")

    # x-variables
    x = {i: m.addVar(vtype=GRB.BINARY, name=f"x[{i}]") for i in range(n)}

    # y-variables
    y = {
        (i, j): m.addVar(vtype=GRB.BINARY, name=f"y[{i},{j}]")
        for i in range(n)
        for j in range(i + 1, n)
    }

    # Capacity
    m.addConstr(sum(weights[i] * x[i] for i in range(n)) <= capacity)

    # Linking
    for i, j in y:
        m.addConstr(y[i, j] <= x[i])
        m.addConstr(y[i, j] <= x[j])

    # Objective
    obj = sum(profits[i] * x[i] for i in range(n)) + sum(
        quad_profits[i][j] * y[(i, j)]
        for i in range(n)
        for j in range(i + 1, n)
    )

    m.setObjective(obj, GRB.MAXIMIZE)
    m.setParam("TimeLimit", 15)
    m.optimize()

    return m, x, m.objVal


# ==============================================================
# 5. Reduced ILP: FORCE a set of items S_forced to x_i = 1
# ==============================================================
def solve_reduced_fix_set(weights, profits, quad_profits, capacity, S_forced):
    n = len(weights)
    S = set(S_forced)

    m = Model("QKP_reduced_forced")

    # x-variables for all items
    x = {i: m.addVar(vtype=GRB.BINARY, name=f"x[{i}]") for i in range(n)}

    # Fix forced items to 1
    for i in S:
        m.addConstr(x[i] == 1)

    # y-variables
    y = {
        (i, j): m.addVar(vtype=GRB.BINARY, name=f"y[{i},{j}]")
        for i in range(n)
        for j in range(i + 1, n)
    }

    # Linking
    for i, j in y:
        m.addConstr(y[i, j] <= x[i])
        m.addConstr(y[i, j] <= x[j])

    # Capacity
    m.addConstr(sum(weights[i] * x[i] for i in range(n)) <= capacity)

    # Objective
    obj = sum(profits[i] * x[i] for i in range(n)) + sum(
        quad_profits[i][j] * y[(i, j)]
        for i in range(n)
        for j in range(i + 1, n)
    )

    m.setObjective(obj, GRB.MAXIMIZE)
    m.setParam("TimeLimit", 15)
    m.optimize()

    return m, x, m.objVal


def solve_reduced_ilp_init(weights, profits, quad_profits, capacity, selected):
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

    if m.status == GRB.OPTIMAL:
        return m, x, m.objVal


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    folder = "InstancesEx1/"
    files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    results = {
        "full_ilp": [],
        "reduced_ilp_forced": [],
        "reduced_ilp_forced_without_constraints": [],
        "greedy": [],
    }
    for fname in files:
        print("\n==============================")
        print("INSTANCE:", fname)
        print("========== ====================")
        filename = os.path.join(folder, fname)
        n, cap, weights, quad = read_instance(filename)
        profits = [quad[i][i] for i in range(n)]

        # -----------------------------
        # Full ILP
        # -----------------------------
        S_opt_model, x_vars, obj_full = solve_ilp(weights, profits, quad, cap)
        S_opt = {i for i in x_vars if x_vars[i].X > 0.5}

        print(f"Full ILP objective: {obj_full}")
        print(f"Optimal items: {sorted(S_opt)}")
        results["full_ilp"].append(obj_full)
        # -----------------------------
        # Forced Reduced ILP Test
        # -----------------------------
        all_items = set(range(n))
        non_opt_items = list(all_items - S_opt)

        K = min(10, len(non_opt_items))
        S_forced = set(non_opt_items[:K])
        print(f"Forced (bad) items: {sorted(S_forced)}")

        _, _, obj_reduced = solve_reduced_fix_set(
            weights, profits, quad, cap, S_forced
        )

        print(f"Reduced ILP objective (forcing bad items): {obj_reduced}")
        results["reduced_ilp_forced"].append(obj_reduced)
        # -----------------------------
        # reduced initial ILP without ocnstraints
        # ------------------------------

        _, _, obj_reduced = solve_reduced_ilp_init(
            weights, profits, quad, cap, S_forced
        )
        print(
            f"Reduced ILP objective without constraints(forcing bad items): {obj_reduced}"
        )
        results["reduced_ilp_forced_without_constraints"].append(obj_reduced)
        # -----------------------------
        # Greedy baseline
        # -----------------------------
        S_greedy = greedy_qkp(weights, profits, quad, cap)
        greedy_value = compute_profit(S_greedy, profits, quad)
        print(f"Greedy objective: {greedy_value}")
        results["greedy"].append(greedy_value)

for method in results:
    avg = sum(results[method]) / len(results[method])
    print(f"\nAverage objective for {method}: {avg}")
