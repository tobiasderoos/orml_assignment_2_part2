import os
import random
import csv
import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
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


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward):
        self.buffer.append((state, action, reward))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class FeatureExtractor:
    def __init__(self):
        self.feature_dim = 12

    def extract(self, n, weights, profits, quad, capacity):
        w = np.array(weights, dtype=np.float32)
        p = np.array(profits, dtype=np.float32)
        Q = np.array(quad, dtype=np.float32)

        pw = p / w
        pw_mean = pw.mean()
        pw_std = pw.std()
        pw_cv = pw_std / pw_mean
        pw_skew = skew(pw)
        gini_pw = self._gini(pw)

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
                gini_pw,
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
        feats = np.nan_to_num(feats, nan=0.0)
        return feats.reshape(1, -1)

    def _gini(self, x):
        x = np.sort(x)
        n = len(x)
        return (2 * np.sum((np.arange(1, n + 1)) * x)) / (n * x.sum()) - (n + 1) / n


class DQNAgent:
    def __init__(
        self,
        feature_dim,
        lr=3e-4,
        epsilon_decay=0.9995,
        epsilon_min=0.1,
        batch=32,
    ):
        self.actions = np.arange(45, 110, 2).tolist()
        self.n_actions = len(self.actions)
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr

        self.batch_size = batch

        self.q_min, self.q_max = -5.0, 10.0
        self.model = self._build_model(feature_dim)

        self.replay = ReplayBuffer(capacity=30000)
        self.warm_up = 40

    def _build_model(self, dim):
        inp = layers.Input(shape=(dim,))
        x = layers.BatchNormalization()(inp)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(self.n_actions, activation="softmax")(x)

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
                [  # episode, loss, epsilon, reward, action, greedy action, q_range, q_mean, stopping, profit, reward_diff
                    "episode",
                    "loss",
                    "epsilon",
                    "reward",
                    "action",
                    "greedy_action",
                    "q_range",
                    "q_mean",
                    "stopping_threshold",
                    "profit",
                    "reward_diff",
                ]
            )

        for ep in tqdm.tqdm(range(n_episodes)):
            file = random.choice(instance_files)
            n, cap, w, Q = read_instance(file)
            p = [Q[i][i] for i in range(n)]

            state = extractor.extract(n, w, p, Q, cap)

            max_abs_feat = float(np.max(np.abs(state)))
            writer.add_scalar("Features/MaxAbs", max_abs_feat, ep)
            writer.add_scalar("Features/MeanAbs", float(np.mean(np.abs(state))), ep)
            writer.add_scalar("Features/Std", float(np.std(state)), ep)

            action_idx, q = agent.act(state)
            q_sa = q[action_idx]

            greedy_idx = int(np.flatnonzero(q == q.max())[-1])
            stopping = actions[action_idx]

            # Baseline greedy (no stopping)
            greedy_full = greedy_qkp(w, p, Q, cap, None)
            greedy_profit = compute_profit(greedy_full, p, Q)
            td_error = 0.0
            true_reward = 0.0
            status = None
            obj = greedy_profit

            # Greedy with stopping threshold
            greedy_sel = greedy_qkp(w, p, Q, cap, stopping)

            remaining = cap - sum(w[i] for i in greedy_sel)
            candidates = [
                i for i in range(n) if i not in greedy_sel and w[i] <= remaining
            ]
            key = (tuple(sorted(greedy_sel)), stopping)
            if not candidates:
                reward = -2.0
                obj = greedy_profit
                true_reward = 0.0
                td_error = 0.0
                reward = -2.0
                # selected_items = greedy_sel
            else:
                if key not in rilp_cache:
                    start = time.time()
                    obj, q_selected, status = solve_reduced_ilp(w, p, Q, cap, greedy_sel)
                    q_selected = [i for i, val in q_selected.items() if val > 0.5]
                    q_selected = set(q_selected)
                    true_reward = (obj / greedy_profit) - 1.0
                    reward = true_reward * 15.0
                    end = time.time()
                    elapsed = end - start

                    if status == GRB.Status.TIME_LIMIT:
                        reward -= 0.5
                    elif status == GRB.Status.INFEASIBLE:
                        reward -= 2.0
                    elif status == GRB.Status.OPTIMAL:
                        reward += 0.25 * ((end - start) / 15.0)
                    reward = np.clip(reward, agent.q_min, agent.q_max)
                    td_error = abs(q_sa - reward)
                    rilp_cache[key] = (
                        obj,
                        true_reward,
                        reward,
                        q_selected,
                        td_error,
                    )
                else:
                    obj, true_reward, reward, q_selected, td_error = rilp_cache[key]

            # add batch sample
            agent.replay.add(state.squeeze(), action_idx, float(reward))

            # when batch full, train
            loss = 0.0
            if len(agent.replay) >= agent.warm_up:
                states, actions_b, rewards_b = agent.replay.sample(batch_size)

                q_preds = agent.model.predict(states, verbose=0)
                targets = q_preds.copy()
                for i, a in enumerate(actions_b):
                    targets[i, a] = rewards_b[i]

                loss = agent.model.train_on_batch(states, q_preds)
                writer.add_scalar("Loss", loss, ep)

                # logging per episode
            q_range = float(q.max() - q.min())
            q_mean = np.mean(q)
            writer.add_scalar("Loss", loss, ep)
            writer.add_scalar("Epsilon", agent.epsilon, ep)
            writer.add_scalar("Reward", reward, ep)
            writer.add_scalar("Action", actions[action_idx], ep)
            writer.add_scalar("GreedyAction", actions[greedy_idx], ep)
            writer.add_scalar("Q/Range", q_range, ep)
            writer.add_scalar("Q/Mean", q_mean, ep)
            writer.add_scalar("Stopping", stopping, ep)
            writer.add_scalar("Profit", true_reward, ep)
            writer.add_scalar("reward_diff", td_error, ep)

            csv_writer.writerow(
                [  # episode instance, loss, epsilon, reward, action, greedy action, q_range, q_mean, stopping, profit, reward_diff
                    ep,
                    loss,
                    agent.epsilon,
                    reward,
                    actions[action_idx],
                    actions[greedy_idx],
                    q_range,
                    q_mean,
                    stopping,
                    true_reward,
                    td_error,
                ]
            )

            f.flush()

            # Store model every 500 episodes
            if (ep + 1) % 250 == 0:
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
    n_episodes = 3000
    batch_size = 32

    extractor = FeatureExtractor()
    agent = DQNAgent(
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
