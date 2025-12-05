def compute_frac_high_state(weights, profits):
    ratios = [profits[i] / weights[i] for i in range(len(weights))]
    n = len(weights)
    avg_ratio = sum(ratios) / n

    if avg_ratio < 2.0:
        ratio_bin = 0
    elif avg_ratio < 2.6:
        ratio_bin = 1
    else:
        ratio_bin = 2

    return ratio_bin


if __name__ == "__main__":
    import os
    from collections import Counter
    from training_q import read_instance

    state_ids = []
    instance_files = [
        os.path.join("InstancesEx1_200", f)
        for f in os.listdir("InstancesEx1")
        if f.endswith(".txt")
    ]

    for fname in instance_files:
        n, c, w, p = read_instance(fname)
        p = [p[i][i] for i in range(n)]
        state_id = compute_state_id(w, p, c)
        state_ids.append(state_id)

    counter = Counter(state_ids)

    print("State distribution:")
    for state in range(15):
        print(f"State {state:2d}: {counter[state]} occurrences")

    # Optioneel: mooi histogram
    import matplotlib.pyplot as plt

    plt.bar(counter.keys(), counter.values())
    plt.xlabel("State ID")
    plt.ylabel("Count")
    plt.title("Distribution of Q-learning states")
    plt.show()
