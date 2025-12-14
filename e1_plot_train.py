import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import entropy

data = pd.read_csv("exc_1_results/train_results.csv")


def smooth_ema(data, alpha=0.1):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def smooth_ema(data, alpha=0.05):
    smoothed = np.zeros_like(data, dtype=float)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


# parameters
window = 100
alpha = 0.05
epsilon_min_episode = data.loc[data["epsilon"] <= 0.1, "episode"].iloc[0]

# ---------------------
# Plot Q - reward diff with EMA
# ---------------------
ema = smooth_ema(data["diff"].values, alpha=alpha)
rolling_std = pd.Series(data["diff"]).rolling(window=window, min_periods=1).std().values

plt.figure()
plt.plot(data["episode"], ema, label="EMA")
plt.fill_between(
    data["episode"],
    ema - rolling_std,
    ema + rolling_std,
    alpha=0.2,
    label="±1 rolling std",
)

plt.xlabel("Episode")
plt.ylabel("Diff")
plt.axvline(
    epsilon_min_episode,
    linestyle="--",
    linewidth=1,
    alpha=0.7,
    color="red",
    label="Min $\epsilon$",
)
plt.axvspan(0, 200, alpha=0.08, color="gray", label="Warm-up")
plt.legend(fontsize=8)
plt.show()

# ---------------------
# Plot Epsilon Decay
# ---------------------
plt.figure()
plt.plot(data["episode"], data["epsilon"], color="blue")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.show()

# ---------------------
# Plot dominant greedy action with rolling window
# ---------------------
window = 25


def action_entropy(x):
    p = np.bincount(x.astype(int))
    p = p[p > 0]
    return entropy(p, base=2)


rolling_entropy = data["greedy_action"].rolling(window).apply(action_entropy, raw=False)
rolling_std = data["greedy_action"].rolling(window).std()
plt.figure()
plt.plot(data["episode"], rolling_entropy, label="Rolling entropy")
plt.axvline(
    epsilon_min_episode,
    linestyle="--",
    linewidth=1,
    alpha=0.7,
    color="red",
    label="Min $\epsilon$",
)
plt.axvspan(0, 200, alpha=0.08, color="gray", label="Warm-up")
plt.xlabel("Episode")
plt.ylabel("Action entropy")
plt.legend(fontsize=8)

plt.show()


# ---------------------
# Plot Q - reward diff scatter
# ---------------------

ema = smooth_ema(data["q_value_range"].values, alpha=alpha)
rolling_std = (
    pd.Series(data["q_value_range"]).rolling(window=window, min_periods=1).std().values
)


plt.figure()
plt.plot(data["episode"], ema, label="EMA")
plt.fill_between(
    data["episode"],
    ema - rolling_std,
    ema + rolling_std,
    alpha=0.2,
    label="±1 rolling std",
)

plt.xlabel("Episode")
plt.ylabel("Q-value range")
plt.axvline(
    epsilon_min_episode,
    linestyle="--",
    linewidth=1,
    alpha=0.7,
    color="red",
    label="Min $\epsilon$",
)
plt.axvspan(0, 200, alpha=0.08, color="gray", label="Warm-up")

plt.legend(fontsize=8)
plt.show()

# ----------------------
# Gaps plot
# ----------------------
gap_12_ema = smooth_ema(data["q_gap_1_2"].values, alpha)
gap_15_ema = smooth_ema(data["q_gap_1_5"].values, alpha)
gap_110_ema = smooth_ema(data["q_gap_1_10"].values, alpha)
gap_12_std = (
    pd.Series(data["q_gap_1_2"]).rolling(window=window, min_periods=1).std().values
)
gap_15_std = (
    pd.Series(data["q_gap_1_5"]).rolling(window=window, min_periods=1).std().values
)
gap_110_std = (
    pd.Series(data["q_gap_1_10"]).rolling(window=window, min_periods=1).std().values
)

plt.figure()

plt.plot(
    data["episode"],
    data["q_gap_1_2"],
    alpha=0.15,
    linewidth=1,
    label="Q gap 1–2 (raw)",
)
plt.plot(
    data["episode"],
    data["q_gap_1_5"],
    alpha=0.15,
    linewidth=1,
    label="Q gap 1–5 (raw)",
)
plt.plot(
    data["episode"],
    data["q_gap_1_10"],
    alpha=0.15,
    linewidth=1,
    label="Q gap 1–10 (raw)",
)

plt.plot(data["episode"], gap_12_ema, linewidth=2, label="Q gap 1–2 (EMA)")
plt.plot(data["episode"], gap_15_ema, linewidth=2, label="Q gap 1–5 (EMA)")
plt.plot(data["episode"], gap_110_ema, linewidth=2, label="Q gap 1–10 (EMA)")

plt.fill_between(
    data["episode"],
    gap_12_ema - gap_12_std,
    gap_12_ema + gap_12_std,
    alpha=0.15,
)

plt.fill_between(
    data["episode"],
    gap_15_ema - gap_15_std,
    gap_15_ema + gap_15_std,
    alpha=0.15,
)

plt.fill_between(
    data["episode"],
    gap_110_ema - gap_110_std,
    gap_110_ema + gap_110_std,
    alpha=0.15,
)

plt.axvline(
    epsilon_min_episode,
    linestyle="--",
    linewidth=1,
    alpha=0.7,
    color="red",
)

plt.xlabel("Episode")
plt.ylabel("Q-gap")
plt.legend(fontsize=8)
plt.axvline(
    epsilon_min_episode,
    linestyle="--",
    linewidth=1,
    alpha=0.7,
    color="red",
    label="Min $\epsilon$",
)
plt.axvspan(0, 200, alpha=0.08, color="gray", label="Warm-up")


plt.show()
