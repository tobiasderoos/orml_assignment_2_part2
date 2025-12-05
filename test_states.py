def compute_frac_high_state(profits, weights):
    ratios = [profits[i] / weights[i] for i in range(len(weights))]
    n = len(weights)
    avg_ratio = sum(ratios) / n

    if avg_ratio < 2.38:
        ratio_bin = 0
    elif avg_ratio < 2.86:
        ratio_bin = 1
    else:
        ratio_bin = 2
    return avg_ratio, ratio_bin


def compute_theoretical_capacity(weights, capacity):
    avg_weight = sum(weights) / len(weights)
    theoretical_capacity = capacity / avg_weight
    return theoretical_capacity


if __name__ == "__main__":
    import os
    from collections import Counter
    from matplotlib import pyplot as plt
    from greedy import read_instance

    import numpy as np

    instance_folder = "InstancesEx1_200"
    state_ids = []
    instance_files = [
        f for f in os.listdir(instance_folder) if f.endswith(".txt")
    ]

    state_ids = []
    avg_ratios = []
    for fname in instance_files:
        n, c, w, p = read_instance(os.path.join(instance_folder, fname))
        p = [p[i][i] for i in range(n)]

        # compute state ID
        avg_ratio, ratio_bin = compute_frac_high_state(p, w)
        state_ids.append(ratio_bin)
        avg_ratios.append(avg_ratio)

plt.figure(figsize=(8, 5))
plt.hist(avg_ratios, bins=40, edgecolor="black")
plt.xlabel("Profit / weight ratio")
plt.ylabel("Frequency")
plt.title("Distribution of average p/w ratios across all instances")
plt.grid(alpha=0.3)
plt.show()

counts = Counter(state_ids)
print(counts)

q_25 = np.percentile(avg_ratios, 25)
q_50 = np.percentile(avg_ratios, 50)
q_75 = np.percentile(avg_ratios, 75)
print(f"25th percentile: {q_25}")
print(f"50th percentile: {q_50}")
print(f"75th percentile: {q_75}")
