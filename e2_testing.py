import numpy as np
import pandas as pd


def compute_profit(S, profits, quad_profits):
    # NOTE S SHOULD BE INDICES NOT BOOL MASK
    S = np.where(S == 1)[0]
    total = sum(profits[i] for i in S)
    total += sum(quad_profits[i][j] for i in S for j in S if i < j)
    return total


def greedy_qkp(
    n,
    weights,
    profits,
    quad,
    capacity,
):
    selected = set()
    remaining_capacity = capacity
    while remaining_capacity > 0:
        candidates = [
            i for i in range(n) if i not in selected and weights[i] <= remaining_capacity
        ]

        if not candidates:
            break

        best_item = max(candidates, key=lambda i: profits[i] / weights[i])
        selected.add(best_item)
        remaining_capacity -= weights[best_item]

    # Compute total profit including quadratic terms
    total_profit = 0
    for i in selected:
        total_profit += profits[i]
        for j in selected:
            if i < j:
                total_profit += quad[i][j]

    return total_profit


if __name__ == "__main__":
    import os
    from e2_training_q import DeepQEnv, DeepQAgent
    from e2_performanceEx2 import read_instance

    # Load training instances
    folder = "InstancesEx2_test"
    instance_files = [
        os.path.join(folder, fname)
        for fname in os.listdir(folder)
        if fname.endswith(".txt")
    ]
    # Load Keras mode
    model = DeepQAgent.load_model("exc_2_model/dqn_qkp_model_ep4000.keras")

    envs = []

    results = []

    for i, f in enumerate(instance_files):
        n, cap, w, q = read_instance(f)
        p = [q[i][i] for i in range(n)]
        envs.append(DeepQEnv(w, p, q, cap, instance_id=i))

        greedy_profit = greedy_qkp(n, w, p, q, cap)
        Q_profit = model.evaluate(envs[i])

        improvement = 100 * (Q_profit - greedy_profit) / abs(greedy_profit)

        results.append(
            {
                "instance": i,
                "greedy_profit": float(greedy_profit),
                "q_profit": float(Q_profit),
                "improvement_pct": float(improvement),
            }
        )

        print(
            f"Instance {i:02d} | Greedy: {greedy_profit:.2f} | "
            f"Q: {Q_profit:.2f} | Improvement: {improvement:.2f}%"
        )

    df = pd.DataFrame(results)
    df.set_index("instance", inplace=True)

    numeric_cols = ["greedy_profit", "q_profit", "improvement_pct"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    summary = pd.DataFrame(
        {
            "metric": ["greedy_profit", "q_profit", "improvement_pct"],
            "min": [df.greedy_profit.min(), df.q_profit.min(), df.improvement_pct.min()],
            "max": [df.greedy_profit.max(), df.q_profit.max(), df.improvement_pct.max()],
            "mean": [
                df.greedy_profit.mean(),
                df.q_profit.mean(),
                df.improvement_pct.mean(),
            ],
            "std": [df.greedy_profit.std(), df.q_profit.std(), df.improvement_pct.std()],
        }
    )

    summary.to_csv("exc_2_results/e2_testing_summary.csv", index=False)
