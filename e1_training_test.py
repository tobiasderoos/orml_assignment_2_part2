import os
import random
import csv
import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from scipy.stats import skew
from datetime import datetime
from gurobipy import GRB
from torch.utils.tensorboard import SummaryWriter

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from e1_testing import (
    greedy_qkp,
    compute_profit,
    solve_reduced_ilp,
)

from e2_performanceEx2 import read_instance


class FeatureExtractor:
    def __init__(self):
        self.feature_dim = 11

    def extract(self, n, weights, profits, quad, capacity):
        w = np.array(weights, dtype=np.float32)
        p = np.array(profits, dtype=np.float32)
        Q = np.array(quad, dtype=np.float32)

        pw = p / w
        pw_mean = pw.mean()
        pw_std = pw.std()
        pw_cv = pw_std / pw_mean
        pw_skew = skew(pw)
        # gini_pw = self._gini(pw)

        pw_iw_corr = np.corrcoef(p, w)[0, 1]

        sorted_idx = np.argsort(pw)[::-1]
        cum_w = np.cumsum(w[sorted_idx])
        m = int(np.searchsorted(cum_w, capacity, side="right"))
        fit_ratio = m / n
        cap_tight = capacity / w.mean()

        pw_sorted = np.sort(pw)[::-1][:m]
        deltas = pw_sorted[:-1] - pw_sorted[1:]

        delta_mean = deltas.mean()
        delta_median = np.median(deltas)
        delta_std = deltas.std()
        delta_cv = delta_std / delta_mean if delta_mean != 0 else 0.0
        delta_skew = skew(deltas)

        greedy_items = sorted_idx[:m]
        quad_inc = []
        for t in range(1, m):
            i = greedy_items[t]
            quad_inc.append(Q[i, greedy_items[:t]].sum() / t)

        quad_inc = np.array(quad_inc)
        quad_corr = np.corrcoef(deltas, quad_inc)[0, 1]
        quad_skew = skew(quad_inc)

        feats = np.array(
            [
                delta_mean,
                delta_median,
                delta_cv,
                delta_skew,
                # gini_pw
                pw_cv,
                pw_skew,
                pw_iw_corr,
                cap_tight,
                fit_ratio,
                quad_corr,
                quad_skew,
            ],
            dtype=np.float32,
        )

        return feats.reshape(1, -1)

    def _gini(self, x):
        x = np.sort(x)
        n = len(x)
        return (2 * np.sum((np.arange(1, n + 1)) * x)) / (n * x.sum()) - (n + 1) / n


class DQNAgent:
    def __init__(
        self,
        n_actions,
        feature_dim,
        lr=3e-4,
        epsilon_decay=0.999,
        epsilon_min=0.1,
        batch=32,
    ):
        self.n_actions = n_actions
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr

        self.batch_size = batch

        self.q_min, self.q_max = -5.0, 10.0
        self.model = self._build_model(feature_dim)

    def _build_model(self, dim):
        inp = layers.Input(shape=(dim,))
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

    def act(self, state, train=True):
        q = self.model.predict(state, verbose=0).flatten()

        # TRAIN MODE
        if train:
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.n_actions), q
            # random tie-break
            greedy = np.random.choice(np.flatnonzero(q == q.max()))
            return greedy, q

        else:
            # deterministic: last index of max Q
            greedy = np.flatnonzero(q == q.max())[-1]
            return greedy, q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(
    agent,
    extractor,
    instance_files,
    actions,
    writer,
    n_episodes=5000,
    batch_size=32,
    store_dir="exc_1_model_new",
):
    os.makedirs(store_dir, exist_ok=True)
    csv_path = os.path.join(store_dir, "train_results.csv")

    rilp_cache = {}

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        csv_writer = csv.writer(f)
        if not file_exists:
            csv_writer.writerow(
                [
                    "episode",
                    "instance",
                    "loss",
                    "epsilon",
                    "reward",
                    "action",
                    "greedy_action",
                    "q_range",
                    "stopping_threshold",
                    "true_reward",
                    "shaped_true",
                    "reward_diff",
                ]
            )
        batch_states, batch_actions, batch_rewards = [], [], []

        for ep in tqdm.tqdm(range(n_episodes)):
            file = random.choice(instance_files)
            n, cap, w, Q = read_instance(file)
            p = [Q[i][i] for i in range(n)]

            state = extractor.extract(n, w, p, Q, cap)
            action_idx, q = agent.act(state)
            greedy_idx = int(np.flatnonzero(q == q.max())[-1])
            stopping = actions[action_idx]

            key = (file, stopping)

            # Baseline greedy (no stopping)
            greedy_full = greedy_qkp(w, p, Q, cap, None)
            greedy_profit = compute_profit(greedy_full, p, Q)

            if key in rilp_cache:
                obj, status, reward = rilp_cache[key]
            else:
                # Greedy with stopping threshold
                greedy_sel = greedy_qkp(w, p, Q, cap, stopping)

                remaining = cap - sum(w[i] for i in greedy_sel)
                candidates = [
                    i for i in range(n) if i not in greedy_sel and w[i] <= remaining
                ]
                if not candidates:
                    reward = -2.0
                else:
                    start = time.time()
                    obj, _, status = solve_reduced_ilp(w, p, Q, cap, greedy_sel)
                    # reward = (
                    #     ((obj / greedy_profit) - 1.0) * 15.0 if obj is not None else -2.0
                    # )

                    true_reward = (obj / greedy_profit) - 1.0
                    reward = true_reward * 15.0
                    end = time.time()
                    if status == GRB.Status.TIME_LIMIT:
                        reward -= 0.5
                    elif status == GRB.Status.INFEASIBLE:
                        reward -= -2.0
                    elif status == GRB.Status.OPTIMAL:
                        reward += 0.25 * ((end - start) / 15.0)
                rilp_cache[key] = (obj, status, reward)

            # Calculate true predicted reward
            shaped_true = true_reward * 15.0
            reward_diff = reward - shaped_true

            # collect batch sample
            batch_states.append(state.squeeze())
            batch_actions.append(action_idx)
            batch_rewards.append(float(reward))

            # when batch full, train
            loss = 0
            if len(batch_states) >= batch_size:
                states = np.array(batch_states, dtype=np.float32)
                q_preds = agent.model.predict(states, verbose=0)

                for i, a in enumerate(batch_actions):
                    q_preds[i, a] = batch_rewards[i]

                loss = agent.model.train_on_batch(states, q_preds)
                writer.add_scalar("Loss", loss, ep)

                batch_states, batch_actions, batch_rewards = [], [], []

                # logging per episode
            q_range = float(q.max() - q.min())
            writer.add_scalar("Reward", reward, ep)
            writer.add_scalar("QRange", q_range, ep)
            writer.add_scalar("Epsilon", agent.epsilon, ep)
            writer.add_scalar("ChosenAction", actions[action_idx], ep)
            writer.add_scalar("GreedyAction", actions[greedy_idx], ep)
            writer.add_scalar("Profit", obj, ep)
            writer.add_scalar("StoppingThreshold", stopping, ep)
            writer.add_scalar("TrueReward", true_reward, ep)
            writer.add_scalar("ShapedRewardBase", shaped_true, ep)
            writer.add_scalar("RewardDiff", reward_diff, ep)

            csv_writer.writerow(
                [
                    ep,
                    file,
                    loss,
                    agent.epsilon,
                    reward,
                    action_idx,
                    greedy_idx,
                    q_range,
                    stopping,
                    true_reward,
                    shaped_true,
                    reward_diff,
                ]
            )

            f.flush()

            # Store model every 500 episodes
            if (ep + 1) % 500 == 0:
                model_path = os.path.join(store_dir, f"dqn_model_ep{ep + 1}.keras")
                agent.model.save(model_path)
                print(f"Saved model to: {model_path}")

            # epsilon decay per episode
            agent.decay_epsilon()

    writer.close()


if __name__ == "__main__":
    # Instances folder
    instance_dir = "InstancesEx1_train"
    instance_files = [
        os.path.join(instance_dir, fname)
        for fname in os.listdir(instance_dir)
        if fname.endswith(".txt")
    ]

    # Actions (thresholds)
    actions = np.arange(45, 110, 2).tolist()

    # Train settings
    store_dir = "exc_1_results_new"
    n_episodes = 5000
    batch_size = 32

    extractor = FeatureExtractor()
    agent = DQNAgent(
        n_actions=len(actions),
        feature_dim=extractor.feature_dim,
        lr=3e-4,
        epsilon_decay=0.999,
        epsilon_min=0.05,
    )

    log_dir = os.path.join(store_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)

    train(
        agent=agent,
        extractor=extractor,
        instance_files=instance_files,
        actions=actions,
        writer=writer,
        n_episodes=n_episodes,
        batch_size=batch_size,
        store_dir=store_dir,
    )

    model_path = os.path.join(store_dir, "dqn_model_final.keras")
    agent.model.save(model_path)
    print(f"Saved model to: {model_path}")
