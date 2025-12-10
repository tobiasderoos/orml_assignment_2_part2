from datetime import datetime
import os
import pickle
import yaml
import numpy as np
import tqdm
from gurobipy import GRB
import time
import shutil
from torch.utils.tensorboard import SummaryWriter
# import hashlib
# import os


import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Q-Learning
# ---------------------------------------------------------------------


class QLearning:
    """
    Q-Learning hyper-heuristic to learn the best stopping criterion.

    States:  Instance characteristics (theoretical capacity bucket)
    Actions: Number of items to add in greedy phase (stopping criterion k)
    Reward:  Normalized quality of reduced ILP solution
    """

    def __init__(
        self,
        instance_files,
        reset_params=False,
        model_name="qlearning_model",
    ):
        shutil.rmtree("logs_exc1", ignore_errors=True)
        os.makedirs("logs_exc1", exist_ok=True)

        os.makedirs("exc_1_model", exist_ok=True)
        # self.full_ilp_cache_dir = "exc_1_cache"
        # os.makedirs(self.full_ilp_cache_dir, exist_ok=True)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f"logs_exc1/run_{run_id}")
        self.reset_params = reset_params
        self.model_name = model_name

        self.instance_files = instance_files

        # Filenames derived automatically
        self.model_pkl = f"{self.model_name}.pkl"
        self.model_yaml = f"{self.model_name}.yaml"

        # Actions: stopping criterion values (number of items to select in greedy)
        # Here: 5, 10, ..., 95
        self.actions = [5, 10, 20, 30, 50]

        # States: based on instance characteristics
        self.n_states = 9

        # Q-table: states x actions
        self.q_table = np.zeros((self.n_states, len(self.actions)))

        # Q-learning
        self.alpha = 0.25  # Learning rate
        self.epsilon = 1.0  # Exploration rate (starts high)
        self.epsilon_decay = 0.9975
        self.epsilon_min = 0.05

        # Load
        self.load_if_exists()

        # Reset

        # Caches
        # self.full_ilp_cache = {}
        self.reduced_ilp_cache = {}

        self.all_episode_rewards = []
        self.all_episode_optimality = []
        self.all_episode_rilp = []
        self.epsilon_history = []
        self.q_value_history = []
        self.diff_history = []

    def _save(self):
        with open(f"{self.model_pkl}", "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "actions": self.actions,
                    "n_states": self.n_states,
                },
                f,
            )
        print(f"Model saved to {self.model_pkl}")

    def _save_params(self, current_episode=0):
        with open(self.model_yaml, "w") as f:
            yaml.dump(
                {
                    "alpha": self.alpha,
                    "epsilon": self.epsilon,
                    "epsilon_decay": self.epsilon_decay,
                    "epsilon_min": self.epsilon_min,
                    "current_episode": current_episode,
                },
                f,
            )
        print(f"Parameters saved to {self.model_yaml}")

    def _load(self):
        with open(self.model_pkl, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.actions = data["actions"]
        self.n_states = data["n_states"]
        print(f"Loaded model from {self.model_pkl}")

    def _load_params(self):
        with open(self.model_yaml, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.alpha = data["alpha"]
        self.epsilon = data["epsilon"]
        self.epsilon_decay = data["epsilon_decay"]
        self.epsilon_min = data["epsilon_min"]
        print(f"Loaded parameters from {self.model_yaml}")

    def load_if_exists(self):
        if os.path.exists(self.model_pkl):
            self._load()
        if (not self.reset_params) and os.path.exists(self.model_yaml):
            self._load_params()

    @staticmethod
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

    @staticmethod
    def compute_theoretical_capacity(weights, capacity):
        avg_weight = sum(weights) / len(weights)
        theoretical_capacity = capacity / avg_weight

        if theoretical_capacity < 95.06:
            return theoretical_capacity, 0
        elif theoretical_capacity < 105.36:
            return theoretical_capacity, 1
        else:
            return theoretical_capacity, 2

    @staticmethod
    def calculate_state(ratio_bin, capacity_bin):
        """Combine 3×3 bins into a single state in [0–8]."""
        return ratio_bin * 3 + capacity_bin

    def get_state(self, weights, profits, capacity):
        _, ratio_bin = self.compute_frac_high_state(profits, weights)
        _, capacity_bin = self.compute_theoretical_capacity(weights, capacity)
        state = self.calculate_state(ratio_bin, capacity_bin)
        return state

    def solve_rilp_cache(
        self,
        filepath,
        weights,
        profits,
        quad,
        capacity,
        fixed_items,
        stopping_criterion,
    ):
        """
        Solve reduced ILP and cache results.
        """
        key = (filepath, stopping_criterion)

        if key in self.reduced_ilp_cache:
            return self.reduced_ilp_cache[key]

        obj_val, x, status = solve_reduced_ilp(
            weights, profits, quad, capacity, fixed_items
        )
        self.reduced_ilp_cache[key] = (obj_val, x, status)
        return obj_val, x, status

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        Returns index into self.actions.
        """
        if np.random.random() < self.epsilon:
            # Explore: random action index
            return np.random.choice(len(self.actions))
        else:
            # Exploit: best action index for this state
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action_idx, reward):
        """
        Q(s,a) <- Q(s,a) + alpha * (r - Q(s,a))
        """
        current_q = self.q_table[state, action_idx]
        self.q_table[state, action_idx] = current_q + self.alpha * (
            reward - current_q
        )
        self.diff_history.append(reward - current_q)

    def compute_reward(self, greedy_profit, rilp_profit):
        if rilp_profit is None:
            return -2.0
        else:
            return ((rilp_profit / greedy_profit) - 1.0) * 100

    def compute_penalty_model_status(self, status):
        if status in [GRB.Status.TIME_LIMIT]:
            return -0.5
        elif status == GRB.Status.INFEASIBLE:
            return -2.0
        else:
            return 0.0

    # Training loop
    def train(self, n_episodes=10):
        print(
            f"Training Q-learning on {len(self.instance_files)} instances for {n_episodes} episodes."
        )
        # TQDM
        for episode in tqdm.tqdm(range(n_episodes)):
            filepath = np.random.choice(self.instance_files)

            # Load instance
            n, capacity, weights, quad = read_instance(filepath)
            profits = [quad[i][i] for i in range(n)]

            state = self.get_state(weights, profits, capacity)

            action_idx = self.choose_action(state)
            stopping_criterion = self.actions[action_idx]

            # Greedy without stopping criterion
            selected_greedy_no_stop = greedy_qkp(
                weights,
                profits,
                quad,
                capacity,
                stopping_criterion=None,
            )
            greedy_profit = compute_profit(
                selected_greedy_no_stop, profits, quad
            )

            # --------
            # Q LEARNING ACTION!
            # --------
            # Greedy
            selected_greedy = greedy_qkp(
                weights,
                profits,
                quad,
                capacity,
                stopping_criterion=stopping_criterion,
            )
            # Check if still capacity left
            remaining_capacity = capacity - sum(
                weights[i] for i in selected_greedy
            )
            candidates = [
                i
                for i in range(n)
                if i not in selected_greedy and weights[i] <= remaining_capacity
            ]
            if not candidates:
                # no more items fit, skip RILP
                print("No remaining capacity after greedy, skipping RILP.")
                # q_val = compute_profit(selected_greedy, profits, quad)
                reward = -1.0
                rilp_obj_val = greedy_profit
                rilp_status = GRB.Status.OPTIMAL
            else:
                print("Solving RILP after greedy selection...")
                # Reduced ILP
                start = time.time()
                rilp_obj_val, _, rilp_status = self.solve_rilp_cache(
                    filepath,
                    weights,
                    profits,
                    quad,
                    capacity,
                    selected_greedy,
                    stopping_criterion,
                )
                end = time.time()
                print(f"RILP: {rilp_obj_val}, greedy: {greedy_profit}")
                print(f"Ratio: {(rilp_obj_val / greedy_profit) - 1}")
                # Rewards and penalties
                reward = self.compute_reward(
                    greedy_profit=greedy_profit,
                    rilp_profit=rilp_obj_val,
                )
                # print(f"Ratio reward: {reward}")

                reward += self.compute_penalty_model_status(rilp_status)

            self.all_episode_rewards.append(reward)
            self.all_episode_optimality.append(
                (rilp_obj_val / greedy_profit) - 1
            )

            self.epsilon_history.append(self.epsilon)
            q_stats = {
                "mean": float(np.mean(self.q_table)),
                "max": float(np.max(self.q_table)),
                "min": float(np.min(self.q_table)),
                "range": float(np.ptp(self.q_table)),
            }

            self.q_value_history.append(q_stats)
            # Print progress every 5 episodes
            if (episode + 1) % 5 == 0:
                print(
                    f"\nEpisode {episode + 1}/{n_episodes} - "
                    f"Avg Reward: {np.mean(self.all_episode_rewards[-5:]):.4f}, "
                    f"Avg Relative Profit: {np.mean(self.all_episode_optimality[-5:]):.4f}, "
                    f"Avg Q-Value Diff: {np.mean(self.diff_history[-5:]):.4f}, "
                    f"Epsilon: {self.epsilon:.4f}"
                )

            self.update_q_table(state, action_idx, reward)

            # TensorBoard Logging
            idx = episode  # clearer

            self.writer.add_scalar(
                "Reward/Episode", self.all_episode_rewards[-1], idx
            )
            self.writer.add_scalar(
                "Optimality/RelativeProfit",
                self.all_episode_optimality[-1],
                idx,
            )

            # Q-table stats
            self.writer.add_scalar("QTable/Mean", q_stats["mean"], idx)
            self.writer.add_scalar("QTable/Max", q_stats["max"], idx)
            self.writer.add_scalar("QTable/Min", q_stats["min"], idx)
            self.writer.add_scalar("QTable/Range", q_stats["range"], idx)

            # Profit diagnostics
            self.writer.add_scalar("Profit/RILP", rilp_obj_val, idx)

            # Exploration
            self.writer.add_scalar("Exploration/Epsilon", self.epsilon, idx)

            # Decay epsilon
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay,
            )
            # Store model every 10 episodes
            if (episode + 1) % 10 == 0:
                # self._save()
                # self._save_params(current_episode=episode + 1)
                self.writer.flush()
        print("\nTraining completed")
        self.writer.close()

        return self


def plot_qlearning_diagnostics(agent, alpha=0.1, save_path=None):
    """
    Diagnostic plots for your Q-learning training run:
    1. Optimality ratio (profit / greedy)
    2. Episode rewards
    3. Q-table statistics across training
    4. Epsilon decay
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ----------------------------------------------------
    # Episode axis
    # ----------------------------------------------------
    episodes = np.arange(len(agent.all_episode_rewards))

    # ----------------------------------------------------
    # 1. Optimality ratio
    # ----------------------------------------------------
    ax1 = axes[0, 0]
    opt = np.array(agent.all_episode_optimality)

    ax1.plot(episodes, opt, alpha=0.3, color="blue", label="raw")
    ax1.plot(
        smooth_ema(opt, alpha), color="blue", linewidth=2, label="smoothed"
    )
    ax1.axhline(1.0, color="red", linestyle="--", label="greedy = 1")

    ax1.set_title("Optimality ratio (RILP / Greedy)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Optimality")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ----------------------------------------------------
    # 2. Episode rewards
    # ----------------------------------------------------
    ax2 = axes[0, 1]
    rewards = np.array(agent.all_episode_rewards)

    ax2.plot(episodes, rewards, alpha=0.3, color="green", label="raw")
    ax2.plot(
        smooth_ema(rewards, alpha), color="green", linewidth=2, label="smoothed"
    )

    ax2.set_title("Episode Rewards")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ----------------------------------------------------
    # 3. Q-table statistics
    # ----------------------------------------------------
    ax3 = axes[1, 0]

    q_means = [q["mean"] for q in agent.q_value_history]
    q_maxs = [q["max"] for q in agent.q_value_history]
    q_mins = [q["min"] for q in agent.q_value_history]

    steps = np.arange(len(q_means))

    ax3.plot(steps, q_means, color="blue", label="mean Q")
    ax3.fill_between(
        steps, q_mins, q_maxs, color="blue", alpha=0.2, label="range"
    )

    ax3.set_title("Q-table statistics")
    ax3.set_xlabel("Training step")
    ax3.set_ylabel("Q value")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # ----------------------------------------------------
    # 4. Epsilon decay
    # ----------------------------------------------------
    ax4 = axes[1, 1]

    eps = np.array(agent.epsilon_history)

    ax4.plot(eps, color="purple", linewidth=2)
    ax4.set_title("Epsilon decay")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Epsilon")
    ax4.grid(alpha=0.3)

    # ----------------------------------------------------
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(
            save_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight"
        )

    plt.show()


def smooth_ema(data, alpha=0.1):
    """
    Exponential moving average smoothing.
    """
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from e1_greedy import (
        read_instance,
        greedy_qkp,
        compute_profit,
        solve_reduced_ilp,
    )

    instance_folder = "InstancesEx1_200"
    instance_files = sorted(
        os.path.join(instance_folder, f)
        for f in os.listdir(instance_folder)
        if os.path.isfile(os.path.join(instance_folder, f))
    )

    agent = QLearning(
        instance_files,
        reset_params=True,
        model_name="exc_1_model/model_qlearning",
    )

    # Train on instances
    agent.train(n_episodes=1000)

    # Save the trained model
    agent._save()

    print("\n" + "-" * 60)
    print("Training complete - Model saved as")
    print("-" * 60)
    plot_qlearning_diagnostics(
        agent, alpha=0.1, save_path="exc_1_plots/qlearning_training.png"
    )
