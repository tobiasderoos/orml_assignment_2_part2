import os
import numpy as np
import matplotlib.pyplot as plt

from e1_greedy import (
    read_instance,
    solve_ilp,
    solve_reduced_ilp,
    greedy_qkp,
)


# ---------------------------------------------------------------------
# RUN STOPPING CRITERION EXPERIMENT ON ALL INSTANCES
# ---------------------------------------------------------------------

instance_folder = f"InstancesEx1/"
instance_files = sorted(
    [f for f in os.listdir(instance_folder) if f.endswith(".txt")]
)

all_gaps = []
max_S_global = 0

for fname in instance_files:
    print(f"\nProcessing {fname}.")
    n, cap, weights, quad = read_instance(os.path.join(instance_folder, fname))
    profits = [quad[i][i] for i in range(n)]

    # --- Full ILP ---
    full_obj, x_full, _ = solve_ilp(weights, profits, quad, cap)
    full_sel = [i for i in range(n) if x_full[i] > 0.5]
    S_max = len(full_sel)

    max_S_global = max(max_S_global, S_max)

    # store gaps for this instance
    gaps_this_instance = []

    for S in range(0, S_max + 1):
        greedy_sel = greedy_qkp(weights, profits, cap, S)
        red_obj, xr, _ = solve_reduced_ilp(
            weights, profits, quad, cap, greedy_sel
        )
        gap = ((full_obj - red_obj) / full_obj) * 100
        gaps_this_instance.append(gap)

    all_gaps.append(gaps_this_instance)

# -------------------------------------------------------------
# PROCESS RESULTS
# -------------------------------------------------------------
for i in range(len(all_gaps)):
    last_val = all_gaps[i][-1]
    while len(all_gaps[i]) < max_S_global + 1:
        all_gaps[i].append(last_val)

gaps_matrix = np.array(all_gaps)

mean_gaps = gaps_matrix.mean(axis=0)
std_gaps = gaps_matrix.std(axis=0)
S_values = np.arange(0, max_S_global + 1)

# ---------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------
plt.figure(figsize=(9, 6))
plt.plot(S_values, mean_gaps, label="Mean gap", color="blue")

plt.fill_between(
    S_values,
    mean_gaps - std_gaps,
    mean_gaps + std_gaps,
    alpha=0.2,
    color="blue",
    label="± 1 Std Dev",
)

plt.xlabel("Stopping criterion items selected (S)")
plt.ylabel("Optimality gap (%)")
plt.grid()
plt.legend()
plt.savefig(f"plots/stopping_criterion_experiment.png", dpi=300)
plt.show()
