import os
import numpy as np
import matplotlib.pyplot as plt

from e1_testing import (
    read_instance,
    solve_ilp,
    solve_reduced_ilp,
    greedy_qkp,
    compute_profit,
)

import tqdm


# ---------------------------------------------------------------------
# RUN STOPPING CRITERION EXPERIMENT ON INSTANCES
# ---------------------------------------------------------------------

instance_folder = f"InstancesEx1_200/"
instance_files = sorted(
    [f for f in os.listdir(instance_folder) if f.endswith(".txt")]
)
instance_files = instance_files[:10]
# Store all gaps
all_gaps = []

start_S = 45
end_S = 95
steps = 2
for fname in instance_files:
    gaps_this_instance = []
    print(f"\nProcessing {fname}.")
    n, cap, weights, quad = read_instance(os.path.join(instance_folder, fname))
    profits = [quad[i][i] for i in range(n)]
    # Greedy with no stopping criterion
    greedy_sel_0 = greedy_qkp(weights, profits, cap, n, stopping_criterion=None)
    profit_greedy_0 = compute_profit(greedy_sel_0, profits, quad)
    print(f"Greedy profit (no stopping criterion): {profit_greedy_0}")

    for S in range(start_S, end_S + 1, steps):
        greedy_stop_sel = greedy_qkp(
            weights, profits, quad, cap, stopping_criterion=S
        )
        rilp_obj, xr, _ = solve_reduced_ilp(
            weights, profits, quad, cap, greedy_stop_sel
        )

        improvement = (rilp_obj / profit_greedy_0) - 1
        print(f"Improvement with S={S}: {improvement * 100:.2f}%")
        gaps_this_instance.append(improvement)

    all_gaps.append(gaps_this_instance)

# # -------------------------------------------------------------
# # PROCESS RESULTS
# # -------------------------------------------------------------
# for i in range(len(all_gaps)):
#     last_val = all_gaps[i][-1]
#     while len(all_gaps[i]) < max_S_global + 1:
#         all_gaps[i].append(last_val)

# gaps_matrix = np.array(all_gaps)

# mean_gaps = gaps_matrix.mean(axis=0)
# std_gaps = gaps_matrix.std(axis=0)
# S_values = np.arange(0, max_S_global + 1)

# # ---------------------------------------------------------------------
# # Plot results
# # ---------------------------------------------------------------------
# plt.figure(figsize=(9, 6))
# plt.plot(S_values, mean_gaps, label="Mean gap", color="blue")

# plt.fill_between(
#     S_values,
#     mean_gaps - std_gaps,
#     mean_gaps + std_gaps,
#     alpha=0.2,
#     color="blue",
#     label="± 1 Std Dev",
# )

# plt.xlabel("Stopping criterion items selected (S)")
# plt.ylabel("Optimality gap (%)")
# plt.grid()
# plt.legend()
# plt.savefig(f"plots/stopping_criterion_experiment.png", dpi=300)
# plt.show()
