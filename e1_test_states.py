def compute_frac_high_state(profits, weights):
    ratios = [profits[i] / weights[i] for i in range(len(weights))]
    n = len(weights)
    avg_ratio = sum(ratios) / n

    if avg_ratio < 2.38:
        ratio_bin = 0
    elif avg_ratio < 2.90:
        ratio_bin = 1
    else:
        ratio_bin = 2
    return avg_ratio, ratio_bin


def compute_theoretical_capacity(weights, capacity):
    avg_weight = sum(weights) / len(weights)
    theoretical_capacity = capacity / avg_weight

    if theoretical_capacity < 95.06:
        return theoretical_capacity, 0
    elif theoretical_capacity < 105.36:
        return theoretical_capacity, 1
    else:
        return theoretical_capacity, 2


def calculate_state(ratio_bin, capacity_bin):
    return ratio_bin * 3 + capacity_bin


if __name__ == "__main__":
    import os
    from collections import Counter
    from matplotlib import pyplot as plt
    from e1_greedy import read_instance

    import numpy as np

    instance_folder = "InstancesEx1_200"
    state_ids = []
    instance_files = [
        f for f in os.listdir(instance_folder) if f.endswith(".txt")
    ]

    ids_high_ratio = []
    avg_ratios = []
    theoretical_capacities = []
    ids_capacity = []
    state_ids
    for fname in instance_files:
        n, c, w, p = read_instance(os.path.join(instance_folder, fname))
        p = [p[i][i] for i in range(n)]

        # compute state ID
        avg_ratio, ratio_bin = compute_frac_high_state(p, w)
        ids_high_ratio.append(ratio_bin)
        avg_ratios.append(avg_ratio)

        theoretical_capacity, capacity_bin = compute_theoretical_capacity(w, c)
        theoretical_capacities.append(theoretical_capacity)
        ids_capacity.append(capacity_bin)

        state_id = calculate_state(ratio_bin, capacity_bin)
        state_ids.append(state_id)

plt.figure(figsize=(8, 5))
plt.hist(avg_ratios, bins=40, edgecolor="black")
plt.xlabel("Profit / weight ratio")
plt.ylabel("Frequency")
plt.title("Distribution of average p/w ratios across all instances")
plt.grid(alpha=0.3)
plt.show()
counts = Counter(ids_high_ratio)
print(counts)

counts_capacity = Counter(ids_capacity)
print(counts_capacity)

counts_state = Counter(state_ids)
print(state_ids)

sorted_counts = dict(sorted(counts_state.items()))
print(sorted_counts)

features = {
    "avg_ratio": avg_ratios,
    "theoretical_capacity": theoretical_capacities,
}


# bins for theoretical capacity
for feature_name, feature_values in features.items():
    print(f"\n{feature_name}:")
    q_33 = np.percentile(feature_values, 33)
    q_67 = np.percentile(feature_values, 67)
    print(f"{feature_name}33rd percentile: {q_33}")
    print(f"{feature_name}67th percentile: {q_67}")
