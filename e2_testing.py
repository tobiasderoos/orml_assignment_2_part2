from e2_performanceEx2 import read_instance

from e2_training_q import DeepQAgent


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
    from e2_training_q import DeepQEnv

    # Load training instances
    folder = "InstancesEx2_test"
    instance_files = [
        os.path.join(folder, fname)
        for fname in os.listdir(folder)
        if fname.endswith(".txt")
    ]
    # Load Keras model
    model = DeepQAgent.load_model("exc_2_model/exc_2_model.keras")

    envs = []

    greedy_profits = []
    q_profits = []

    for i, f in enumerate(instance_files):
        n, cap, w, q = read_instance(f)
        p = [q[i][i] for i in range(n)]
        envs.append(DeepQEnv(w, p, q, cap, instance_id=i))

        greedy_profit = greedy_qkp(n, w, p, q, cap)
        Q_profit = model.evaluate(envs[i])

        greedy_profits.append(greedy_profit)
        q_profits.append(Q_profit)
        improvement = 100 * (Q_profit - greedy_profit) / abs(greedy_profit)

        print(
            f"Instance {i:02d} | Greedy: {greedy_profit:.2f} | Q: {Q_profit:.2f}, Improvement: {improvement:.2f}%"
        )
