import os
import yaml
import pickle
import random
import csv
import shutil
import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from scipy.stats import skew
from scipy.stats import kendalltau
from datetime import datetime
from gurobipy import GRB
from torch.utils.tensorboard import SummaryWriter

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from collections import deque

from e1_testing import (
    read_instance,
    greedy_qkp,
    compute_profit,
    solve_reduced_ilp,
)


# EMA smoothing for plots
def smooth_ema(data, alpha=0.1):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


class QLearning:
    """
    Neural hyper-heuristic for choosing the greedy stopping threshold in QKP.
    """

    def __init__(
        self,
        instance_files,
        reset_params=True,
        model_name="exc1_qkeras",
    ):
        # Logging dirs
        shutil.rmtree("logs_exc1", ignore_errors=True)
        os.makedirs("logs_exc1", exist_ok=True)
        os.makedirs("exc_1_model", exist_ok=True)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f"logs_exc1/run_{run_id}")

        self.instance_files = instance_files
        self.reset_params = reset_params
        self.model_name = model_name
        self.model_pt = f"{self.model_name}.keras"
        self.model_yaml = f"{self.model_name}.yaml"

        # Actions
        self.actions = np.arange(45, 110, 2).tolist()
        self.n_actions = len(self.actions)

        # Hyperparameters
        self.lr = 3e-4
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1

        self.replay_buffer = deque(maxlen=4000)
        self.batch_size = 64
        self.warmup = 200

        self.q_min = -5.0
        self.q_max = 10.0

        # Build model
        self.feature_dim = 12

        self.model = self.build_model()

        self.load_if_exists()

        # Caches & logs
        self.reduced_ilp_cache = {}
        # Convergence metrics
        self.all_episode_rewards = []
        self.action_history = []
        self.exploited_action_history = []
        self.q_value_history = []
        self.epsilon_history = []
        self.reward_q_diff = []

    # Keras model
    def build_model(self):
        inp = layers.Input(shape=(self.feature_dim,))
        x = layers.BatchNormalization()(inp)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(self.n_actions, activation="linear")(x)

        model = models.Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=optimizers.Adam(self.lr, clipnorm=1.0),
            loss="huber",
        )
        print(model.summary())
        return model

    # Save / Load
    def _save(self):
        self.model.save(self.model_pt)
        print(f"Saved model to {self.model_pt}")

    def _save_params(self, episode=0):
        with open(self.model_yaml, "w") as f:
            yaml.dump(
                {
                    "lr": self.lr,
                    "epsilon": self.epsilon,
                    "epsilon_decay": self.epsilon_decay,
                    "epsilon_min": self.epsilon_min,
                    "episode": episode,
                },
                f,
            )
        print(f"Saved params to {self.model_yaml}")

    def _load(self):
        self.model = models.load_model(self.model_pt)
        print(f"Loaded model from {self.model_pt}")

    def _load_params(self):
        with open(self.model_yaml, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.lr = data.get("lr", self.lr)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.epsilon_decay = data.get("epsilon_decay", self.epsilon_decay)
        self.epsilon_min = data.get("epsilon_min", self.epsilon_min)

    def load_if_exists(self):
        if os.path.exists(self.model_pt):
            self._load()
        if (not self.reset_params) and os.path.exists(self.model_yaml):
            self._load_params()

    # Feature engineering

    def extract_features(self, n, weights, profits, quad, capacity):
        """feature engineering"""

        w = np.array(weights, dtype=np.float32)
        p = np.array(profits, dtype=np.float32)
        Q = np.array(quad, dtype=np.float32)

        # Profit/weight ratio
        pw = p / w
        pw_mean = pw.mean()
        pw_std = pw.std()
        pw_cv = pw_std / pw_mean
        pw_skew = skew(pw)
        gini_pw = self._calculate_gini(pw)

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

    def _calculate_gini(self, array):
        """Calculate Gini coefficient of a numpy array."""
        array = np.sort(array.flatten())
        n = array.shape[0]
        cumulative_values = np.cumsum(array)
        gini = (2 * np.sum((np.arange(1, n + 1)) * array)) / (
            n * cumulative_values[-1]
        ) - (n + 1) / n
        return gini

    # Reduced ILP caching
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
        key = (filepath, stopping_criterion, tuple(sorted(fixed_items)))

        if key in self.reduced_ilp_cache:
            return self.reduced_ilp_cache[key]

        obj, x, status = solve_reduced_ilp(
            weights, profits, quad, capacity, fixed_items
        )
        self.reduced_ilp_cache[key] = (obj, x, status)
        return obj, x, status

    # Reward computation
    def compute_reward(self, greedy_profit, rilp_profit):
        if rilp_profit is None:
            return -2.0
        return ((rilp_profit / greedy_profit) - 1.0) * 15.0

    def compute_penalty(self, status):
        if status == GRB.Status.TIME_LIMIT:
            return -0.5
        elif status == GRB.Status.INFEASIBLE:
            return -2.0
        elif status == GRB.Status.OPTIMAL:
            return 0.0
        return 0.0

    def compute_time_bonus(self, time, status):
        if status == GRB.Status.OPTIMAL:
            return 0.25 * (time / 15.0)  # slow = higher bonus
        return 0.0

    def pick_random_best_action(self, q_values):
        """Pick randomly among the best actions."""
        candidates = np.flatnonzero(q_values == q_values.max())
        return np.random.choice(candidates)

    def pick_best_action(self, q_values):
        """Pick the best action (last in case of ties)."""
        candidates = np.flatnonzero(q_values == q_values.max())
        return candidates[-1]

    # Training
    def train(self, n_episodes=500):
        print(f"Training for {n_episodes} episodes.")

        for ep in tqdm.tqdm(range(n_episodes)):
            filepath = np.random.choice(self.instance_files)

            # Load instance
            n, capacity, weights, quad = read_instance(filepath)
            profits = [quad[i][i] for i in range(n)]

            feats = self.extract_features(n, weights, profits, quad, capacity)
            q_values = self.model.predict(feats, verbose=0).flatten()
            greedy_action_idx = self.pick_random_best_action(q_values)

            # ε-greedy selection
            if np.random.rand() < self.epsilon:
                action_idx = np.random.choice(self.n_actions)
            else:
                action_idx = greedy_action_idx

            stopping = self.actions[action_idx]
            best_possible_action = self.pick_best_action(q_values)

            self.action_history.append(action_idx)
            self.exploited_action_history.append(greedy_action_idx)

            # Greedy baseline
            greedy_full = greedy_qkp(weights, profits, quad, capacity, None)
            greedy_profit = compute_profit(greedy_full, profits, quad)

            # Greedy with stopping
            greedy_sel = greedy_qkp(weights, profits, quad, capacity, stopping)

            remaining = capacity - sum(weights[i] for i in greedy_sel)
            candidates = [
                i
                for i in range(n)
                if i not in greedy_sel and weights[i] <= remaining
            ]

            if not candidates:
                reward = -2.0
                rilp_obj = greedy_profit
            else:
                start = time.time()
                rilp_obj, _, status = self.solve_rilp_cache(
                    filepath,
                    weights,
                    profits,
                    quad,
                    capacity,
                    greedy_sel,
                    stopping,
                )
                end = time.time()
                elapsed = end - start
                reward = self.compute_reward(greedy_profit, rilp_obj)
                reward += self.compute_penalty(status)
                reward += self.compute_time_bonus(elapsed, status)
            reward = np.clip(reward, self.q_min, self.q_max)
            self.replay_buffer.append((feats.squeeze(), action_idx, reward))

            # train from replay buffer
            diff = 0.0

            if ep > self.warmup:
                batch = random.sample(self.replay_buffer, self.batch_size)

                # Batch predict for efficiency
                states = np.array([s for s, a, r in batch])
                q_preds = self.model.predict(states, verbose=0)

                for i, (s, a, r) in enumerate(batch):
                    q_preds[i, a] = r  # Terminal state: target is just reward
                q_preds = np.clip(q_preds, self.q_min, self.q_max)

                self.model.train_on_batch(states, q_preds)

            # Calculate diff for current episode
            diff = reward - q_values[action_idx]
            self.reward_q_diff.append(diff)

            self.all_episode_rewards.append(reward)
            self.epsilon_history.append(self.epsilon)

            # Save Q-range metric
            self.q_value_history.append(q_values.max() - q_values.min())

            # TensorBoard logging
            self.writer.add_scalar("Reward", reward, ep)
            self.writer.add_scalar("ChosenAction", action_idx, ep)
            self.writer.add_scalar("GreedyAction", best_possible_action, ep)
            self.writer.add_scalar("RewardQDiff", diff, ep)
            self.writer.add_scalar(
                "QValueRange", q_values.max() - q_values.min(), ep
            )
            self.writer.add_scalar("Epsilon", self.epsilon, ep)

            # Update epsilon
            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay
            )
            # Write results
            file_path = "exc_1_model/results.csv"
            file_exists = os.path.isfile(file_path)

            with open(file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(
                        [
                            "episode",
                            "instance",
                            "epsilon",
                            "reward",
                            "diff",
                            "best_action",
                        ]
                    )
                writer.writerow(
                    [
                        ep,
                        filepath,
                        self.epsilon,
                        reward,
                        diff,
                        best_possible_action,
                    ]
                )

        print("\nTraining complete.")
        self.writer.close()
        self._save()
        self._save_params()

    def evaluate_instance(self, n, capacity, weights, quad):
        profits = [quad[i][i] for i in range(n)]
        feats = self.extract_features(n, weights, profits, quad, capacity)

        # Predict Q-values and pick greedy action
        q_values = self.model.predict(feats, verbose=0).flatten()
        best_action_idx = self.pick_best_action(q_values)
        stopping = self.actions[best_action_idx]

        # Greedy with stopping threshold
        greedy_sel = greedy_qkp(weights, profits, quad, capacity, stopping)
        remaining = capacity - sum(weights[i] for i in greedy_sel)

        candidates = [
            i
            for i in range(n)
            if i not in greedy_sel and weights[i] <= remaining
        ]

        if not candidates:
            rilp_profit = compute_profit(greedy_sel, profits, quad)
            status = "No candidates"
        else:
            rilp_profit, _, status = self.solve_rilp_cache(
                None,
                weights,
                profits,
                quad,
                capacity,
                greedy_sel,
                stopping,
            )
        result = {
            "chosen_threshold": stopping,
            "rilp_profit": rilp_profit,
            "status": status,
            "q_values": q_values.tolist(),
        }
        return result


if __name__ == "__main__":
    instance_folder = "InstancesEx1_200"
    instance_files = sorted(
        os.path.join(instance_folder, f)
        for f in os.listdir(instance_folder)
        if os.path.isfile(os.path.join(instance_folder, f))
    )
    instance_files = instance_files[:50]

    agent = QLearning(
        instance_files,
        reset_params=True,
        model_name="exc_1_model/qkeras_model",
    )

    agent.train(n_episodes=6000)

    test_file = test_files[0]
    n, capacity, weights, quad = read_instance(test_file)

    result = agent.evaluate_instance(n, capacity, weights, quad)
    print(result)
