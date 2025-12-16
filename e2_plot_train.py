import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def smooth_ema(data, alpha=0.1):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


df_episodes = pd.read_csv("exc_2_results/train_episode_results.csv")
df_steps = pd.read_csv("exc_2_results/train_step_results.csv")

df_episodes.describe()
df_steps.describe()

window = 500
alpha = 0.2
# ----------------------------------------------
# Plot TD difference mean with EMA and rolling std
# ----------------------------------------------

y = df_episodes["td_diff_mean"].values
x = df_episodes["episode"].values

ema = smooth_ema(y, alpha=alpha)
rolling_std = pd.Series(y).rolling(window=window, min_periods=1).std().values

plt.figure(figsize=(8, 4))
plt.plot(x, ema, label="EMA (td_diff_mean)")
plt.fill_between(
    x,
    ema - rolling_std,
    ema + rolling_std,
    alpha=0.2,
    label="±1 rolling std",
)
plt.axvspan(0, 20, alpha=0.08, color="gray", label="Warm-up")
plt.xlabel("Episode")
plt.ylabel("TD difference (mean)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# ----------------------------------------------
# Plot network difference
# ----------------------------------------------

y = df_episodes["network_diff"].values
x = df_episodes["episode"].values

ema = smooth_ema(y, alpha=alpha)
rolling_std = pd.Series(y).rolling(window=window, min_periods=1).std().values

plt.figure(figsize=(8, 4))

plt.plot(x, y, alpha=0.1, linewidth=0.6, label="Raw")
plt.plot(x, ema, linewidth=2, label="EMA (network_diff)")

plt.fill_between(
    x,
    ema - rolling_std,
    ema + rolling_std,
    alpha=0.2,
    label="±1 rolling std",
)

plt.axvspan(0, 20, alpha=0.08, color="gray", label="Warm-up")
plt.xlabel("Episode")
plt.ylabel("Network difference")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# ----------------------------------------------
# reward
# ----------------------------------------------
y = df_episodes["total_reward"].values
x = df_episodes["episode"].values

ema = smooth_ema(y, alpha=alpha)
rolling_std = pd.Series(y).rolling(window=window, min_periods=1).std().values

plt.figure(figsize=(8, 4))

plt.plot(x, y, alpha=0.1, linewidth=0.6, label="Raw")
plt.plot(x, ema, linewidth=2, label="EMA (total_reward)")

plt.fill_between(
    x,
    ema - rolling_std,
    ema + rolling_std,
    alpha=0.2,
    label="±1 rolling std",
)

plt.axvspan(0, 20, alpha=0.08, color="gray", label="Warm-up")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Episode reward")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()


# ----------------------------------------------
# profit
# ----------------------------------------------

y = df_episodes["final_profit"].values
x = df_episodes["episode"].values

ema = smooth_ema(y, alpha=alpha)
rolling_std = pd.Series(y).rolling(window=window, min_periods=1).std().values

plt.figure(figsize=(8, 4))

plt.plot(x, y, alpha=0.5, linewidth=0.6, label="Raw")
plt.plot(x, ema, linewidth=2, label="EMA (total profit)")

plt.fill_between(
    x,
    ema - rolling_std,
    ema + rolling_std,
    alpha=0.2,
    label="±1 rolling std",
)

plt.axvspan(0, 20, alpha=0.08, color="gray", label="Warm-up")
plt.xlabel("Episode")
plt.ylabel("Total profit")
plt.title("Episode profit")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# ----------------------------------------------
# remaining capacity
# ----------------------------------------------
y = df_episodes["remaining_capacity"].values
x = df_episodes["episode"].values

ema = smooth_ema(y, alpha=alpha)
rolling_std = pd.Series(y).rolling(window=window, min_periods=1).std().values

plt.figure(figsize=(8, 4))

plt.plot(x, y, alpha=0.05, linewidth=0.5, label="Raw")
plt.plot(x, ema, linewidth=2, label="EMA (remaining_capacity)")

plt.fill_between(
    x,
    ema - rolling_std,
    ema + rolling_std,
    alpha=0.2,
    label="±1 rolling std",
)

plt.axvspan(0, 20, alpha=0.08, color="gray", label="Warm-up")
plt.xlabel("Episode")
plt.ylabel("Remaining capacity")
plt.title("Unused capacity per episode")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()


# plt.figure(figsize=(8, 4))
# plt.plot(df_episodes["episode"], df_episodes["td_diff_mean"], label="Mean")
# plt.plot(df_episodes["episode"], df_episodes["td_diff_max"], label="Max")
# plt.plot(df_episodes["episode"], df_episodes["td_diff_min"], label="Min")
# plt.xlabel("Episode")
# plt.ylabel("TD difference")
# plt.title("TD error statistics")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 4))
# plt.plot(df_episodes["episode"], df_episodes["network_diff"])
# plt.xlabel("Episode")
# plt.ylabel("Network diff")
# plt.title("Policy change magnitude")
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(8, 4))
# plt.plot(df_episodes["episode"], df_episodes["lost_opportunities"])
# plt.xlabel("Episode")
# plt.ylabel("Lost opportunities")
# plt.title("Missed opportunities per episode")
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 4))
# plt.plot(df_episodes["episode"], df_episodes["n_penalties"], label="Penalties")
# plt.plot(df_episodes["episode"], df_episodes["n_bonuses"], label="Bonuses")
# plt.xlabel("Episode")
# plt.ylabel("Count")
# plt.title("Penalties and bonuses")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 4))
# plt.plot(df_episodes["episode"], df_episodes["remaining_capacity"])
# plt.xlabel("Episode")
# plt.ylabel("Remaining capacity")
# plt.title("Unused capacity per episode")
# plt.tight_layout()
# plt.show()
