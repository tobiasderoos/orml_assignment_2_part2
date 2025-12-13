import os
from gurobipy import Model, GRB
import numpy as np
import random


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
    quad,
    capacity,
    stopping_criterion=None,
):
    """
    Greedy heuristic for the 0-1 Quadratic Knapsack Problem (QKP),
    but using only linear profits for item selection.

    # NOTE: also consider quadratic profits in marginal profit calculation

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
    constant_profits = sum(profits[i] for i in S)  # already selected items
    # constant profits from newly selected items
    linear_profit = sum(
        profits[i] * x[i] for i in R
    )  # Variable lineair profits
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
    obj = m.objVal
    return obj, x_updated, m.status


def solve_q(weights, profits, quad, capacity, fixed_items, q_table, state):
    """
    Solve reduced ILP using Q-table to fix items.

    Args:
        weights: list w_i
        profits: list p_i
        quad: matrix p_ij
        capacity: knapsack capacity
        fixed_items: set of items chosen by greedy (S)
        q_table: Q-table for RL agent
        state: current state for RL agent

    Returns:
        obj_val, x, status
    """


# ---------------------------------------------------------------------
# Run script to compare greedy, ILP, reduced ILP, and RL
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from e1_training import QLearning
    import pandas as pd
    import random
    import time
    from tqdm import tqdm

    agent = QLearning(
        instance_files=[],
        reset_params=False,
        model_name="exc_1_model/qkeras_model",
    )
    all_results = []

    for dataset in ["train", "test"]:
        print(f"\n--- DATASET: {dataset.upper()} ---\n")
        instance_folder = f"InstancesEx1_{dataset}/"
        instance_files = [
            f for f in os.listdir(instance_folder) if f.endswith(".txt")
        ]

        instance_files = random.sample(instance_files, 5)

        for fname in tqdm(instance_files):
            filepath = os.path.join(instance_folder, fname)

            n, cap, weights, quad = read_instance(filepath)
            profits = [quad[i][i] for i in range(n)]

            # ---------------------------
            # Greedy
            # ---------------------------
            t0 = time.time()
            S_greedy = greedy_qkp(weights, profits, quad, cap)
            greedy_time = time.time() - t0
            greedy_profit = compute_profit(S_greedy, profits, quad)

            # ---------------------------
            # Full ILP
            # ---------------------------
            t0 = time.time()
            ilp_profit, x_ilp, ilp_status = solve_ilp(
                weights, profits, quad, cap
            )
            ilp_time = time.time() - t0

            # ---------------------------
            # RL reduced ILP
            # ---------------------------
            t0 = time.time()
            rl_result = agent.evaluate_instance(n, cap, weights, quad)
            rl_time = time.time() - t0

            rl_profit = rl_result["rilp_profit"]
            rl_stopping = rl_result["chosen_threshold"]

            # Calculate relative improvements
            rel_improvement_greedy = (
                (rl_profit - greedy_profit) / greedy_profit * 100
            )
            rel_improvement_full = (rl_profit - ilp_profit) / ilp_profit * 100
            combined_score = (
                (rl_profit - greedy_profit) / (ilp_profit - greedy_profit) * 100
            )

            # ---------------------------
            # Storing results per type of model
            # ---------------------------
            all_results.append(
                {
                    "dataset": dataset,
                    "instance": fname,
                    "greedy_profit": greedy_profit,
                    "rl_profit": rl_profit,
                    "ilp_profit": ilp_profit,
                    "rel_improvement_greedy": round(rel_improvement_greedy, 2),
                    "rel_improvement_full": round(rel_improvement_full, 2)
                    if ilp_profit is not None
                    else 100.0,
                    "combined_score": round(combi, 2),
                    "greedy_time": greedy_time,
                    "ilp_time": ilp_time,
                    "rl_time": rl_time,
                    "rl_stopping": rl_stopping,
                }
            )
    df_results = pd.DataFrame(all_results)
    print("\n--- SUMMARY RESULTS ---\n")
    print(df_results)
    df_results.to_csv("exc_1_results/train_test_results.csv", index=False)
