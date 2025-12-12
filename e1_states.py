import os
import numpy as np
import pandas as pd
from scipy.stats import skew

from e1_training import read_instance  # your loader


def calculate_gini(x: np.ndarray) -> float:
    """Gini coefficient calculation"""
    array = x.flatten()
    n = array.shape[0]
    cumulative_values = np.cumsum(array)
    gini = (2 * np.sum((np.arange(1, n + 1)) * array)) / (
        n * cumulative_values[-1]
    ) - (n + 1) / n
    return gini


def create_features(n, weights, profits, quad, capacity):
    """feature engineering"""

    w = np.array(weights, dtype=np.float32)
    p = np.array(profits, dtype=np.float32)
    Q = np.array(quad, dtype=np.float32)

    # Upper triangle (excluding diagonal)
    Q_upper = np.triu(Q, k=1)
    q_vals = Q_upper[np.triu_indices(n, k=1)]

    # Profit/weight ratio
    pw = p / w
    pw_mean = pw.mean()
    pw_std = pw.std()
    pw_cv = pw_std / pw_mean
    pw_skew = skew(pw)
    gini_pw = calculate_gini(pw)

    # Correlation profit vs weight
    pw_iw_corr = np.corrcoef(p, w)[0, 1]

    # Greedy packing
    sorted_indices = np.argsort(pw)[::-1]
    cumulative_weight = np.cumsum(w[sorted_indices])
    items_that_fit = int(
        np.searchsorted(cumulative_weight, capacity, side="right")
    )
    fit_ratio = items_that_fit / n
    cap_tight_mean = capacity / w.mean()

    # Greedy margin statistics
    m = items_that_fit
    pw_sorted = np.sort(pw)[::-1][:m]

    deltas = pw_sorted[:-1] - pw_sorted[1:]
    delta_mean = deltas.mean()
    delta_median = np.median(deltas)
    delta_std = deltas.std()
    delta_cv = delta_std / delta_mean if delta_mean != 0 else 0.0
    delta_skew = skew(deltas)

    # quadratic interaction effects basedd on greedy
    greedy_items = sorted_indices[:m]
    quad_increments = []

    for t in range(1, m):
        i_t = greedy_items[t]
        prev_items = greedy_items[:t]
        quad_sum = Q[i_t, prev_items].sum()
        quad_increments.append(quad_sum / t)

    quad_increments = np.array(quad_increments)
    quad_alignment_corr = np.corrcoef(deltas, quad_increments)[0, 1]
    quad_increment_skew = skew(quad_increments)
    feats = np.array(
        [
            delta_mean,
            delta_median,
            delta_cv,
            delta_skew,
            gini_pw,
            pw_cv,
            pw_skew,
            pw_iw_corr,
            cap_tight_mean,
            fit_ratio,
            quad_alignment_corr,
            quad_increment_skew,
        ],
        dtype=np.float32,
    )
    return feats.reshape(1, -1)


if __name__ == "__main__":
    instance_folder = "InstancesEx1_200"
    instance_files = sorted(
        os.path.join(instance_folder, f)
        for f in os.listdir(instance_folder)
        if os.path.isfile(os.path.join(instance_folder, f))
    )[:100]

    feature_names = [
        "greedy_delta_mean",
        "greedy_delta_median",
        "greedy_delta_cv",
        "greedy_delta_skew",
        "gini_pw",
        "pw_cv",
        "pw_skew",
        "pw_iw_corr",
        "cap_tight_mean",
        "fit_ratio",
        "quad_alignment_corr",
        "quad_increment_skew",
    ]

    rows = []
    for fname in instance_files:
        n, c, w, q = read_instance(fname)
        p = [q[i][i] for i in range(n)]

        feats = create_features(n, w, p, q, c).ravel()

        row = {"instance": fname}
        row.update(
            {
                feature_names[i]: float(feats[i])
                for i in range(len(feature_names))
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows).set_index("instance")

    print(df.head())

    # overall summary statistics across 100 instances
    print("\nAggregate summary over 100 instances:")
    print(df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])
