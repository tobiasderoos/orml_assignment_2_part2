import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
rolling_std = (
    pd.Series(data["diff"]).rolling(window=window, min_periods=1).std().values
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
plt.ylabel("Diff")
plt.axvline(
    epsilon_min_episode, linestyle="--", linewidth=1, alpha=0.7, color="red"
)

plt.legend()
plt.show()

# ---------------------
# Plot Epsilon Decay
# ---------------------
plt.figure()
plt.plot(data["episode"], data["epsilon"], color="orange")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.show()

# ---------------------
# Plot dominant greedy action with rolling window
# ---------------------
rolling_mode = (
    data["best_action"]
    .rolling(window)
    .apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
)

plt.figure()
plt.step(data["episode"], rolling_mode, where="post")
plt.xlabel("Episode")
plt.axvline(
    epsilon_min_episode, linestyle="--", linewidth=1, alpha=0.7, color="red"
)

plt.ylabel("Dominant greedy action")
plt.show()
