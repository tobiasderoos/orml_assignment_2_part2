import os
import numpy as np
import matplotlib.pyplot as plt

from e1_testing import (
    solve_ilp,
    solve_reduced_ilp,
    greedy_qkp,
    compute_profit,
)

from e1_performance import read_instance
import tqdm


# ---------------------------------------------------------------------
# RUN STOPPING CRITERION EXPERIMENT ON INSTANCES
# ---------------------------------------------------------------------

instance_folder = f"InstancesEx1_200/"
instance_files = sorted(
    [f for f in os.listdir(instance_folder) if f.endswith(".txt")]
)
instance_files = instance_files[:50]

# Store all gaps
all_gaps = []

start_S = 45
end_S = 95
steps = 2
range_list = [40, 50, 60, 70, 80, 90, 100, 110, 120]
for fname in instance_files:
    gaps_this_instance = []
    print(f"\nProcessing {fname}.")
    n, cap, w, p = read_instance(os.path.join(instance_folder, fname))
    profits = [p[i][i] for i in range(n)]
    # Greedy with no stopping criterion
    greedy_sel_0 = greedy_qkp(
        weights=w,
        profits=profits,
        quad=p,
        capacity=cap,
        stopping_criterion=None,
    )
    profit_greedy_0 = compute_profit(greedy_sel_0, profits, p)
    print(f"Greedy profit (no stopping criterion): {profit_greedy_0}")
    print(f"Selected items: {len(greedy_sel_0)}")
    for S in range_list:
        if S >= len(greedy_sel_0):
            print(
                f"S={S} exceeds number of items selected ({len(greedy_sel_0)}). Skipping."
            )
            break
        greedy_stop_sel = greedy_qkp(
            weights=w,
            profits=profits,
            quad=p,
            capacity=cap,
            stopping_criterion=S,
        )
        rilp_obj, xr, _ = solve_reduced_ilp(
            weights=w,
            profits=profits,
            quad_profits=p,
            capacity=cap,
            selected=greedy_stop_sel,
        )

        improvement = (rilp_obj / profit_greedy_0) - 1
        print(f"Improvement with S={S}: {improvement:.2f}%")
        gaps_this_instance.append(improvement)

    all_gaps.append(gaps_this_instance)

# Convert all_gaps to a 2D array with NaNs for missing values
num_instances = len(all_gaps)
num_S = len(range_list)

gaps_matrix = np.full((num_instances, num_S), np.nan)

for i, gaps in enumerate(all_gaps):
    gaps_matrix[i, : len(gaps)] = gaps

# Compute differences along S (delta improvement)
# diff[:, j] = gaps[:, j] - gaps[:, j-1]
diff_matrix = np.diff(gaps_matrix, axis=1)
S_diff = range_list[1:]  # x-axis for differences

mean_diff = np.nanmean(diff_matrix, axis=0)
std_diff = np.nanstd(diff_matrix, axis=0)

plt.figure(figsize=(10, 6))

plt.plot(
    S_diff, mean_diff, label="Mean difference in improvement", color="blue"
)
plt.fill_between(
    S_diff,
    mean_diff - std_diff,
    mean_diff + std_diff,
    color="blue",
    alpha=0.25,
    label="± 1 Std Dev",
)

plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel("Stopping criterion S")
plt.ylabel("Mean change in improvement (Δ)")
plt.grid(True)
plt.legend()
plt.savefig("exc_1_plots/stopping_criterion_analysis.pdf", dpi=300)
plt.savefig("exc_1_plots/stopping_criterion_analysis.png", dpi=300)
plt.show()
