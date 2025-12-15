import datetime
import os
import math
import random
from collections import deque

import csv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tqdm

from e2_performanceEx2 import read_instance
from e2_testing import compute_profit

from tensorflow.summary import create_file_writer


class DeepQEnv(gym.Env):
    """
    Gymnasium environment for the quadratic knapsack/QPS problem.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        weights,
        profits,
        quad,
        capacity,
        instance_id=0,
    ):
        super().__init__()
        self.weights = np.array(weights, dtype=np.float32)
        self.profits = np.array(profits, dtype=np.float32)
        self.quad = np.array(quad, dtype=np.float32)

        ratio = self.profits / self.weights
        self.order = np.argsort(ratio)[::-1]  # descending
        self.weights = self.weights[self.order]
        self.profits = self.profits[self.order]
        self.quad = self.quad[np.ix_(self.order, self.order)]

        self.capacity = float(capacity)
        self.n = len(weights)
        self.instance_id = instance_id

        # Normalization constants
        self.max_w = float(np.max(self.weights))
        self.max_p = float(np.max(self.profits))
        self.max_q = float(np.max(np.triu(self.quad, k=1)))
        self.max_linear = float(np.sum(self.profits))
        self.max_gain = float(np.max(self.profits + np.sum(self.quad, axis=1)))

        # Observation space size and action space
        self.obs_dim = 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self.selected = None
        self.remaining_capacity = None
        self.current_profit = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.selected = np.zeros(self.n, dtype=np.int32)
        self.remaining_capacity = self.capacity
        self.current_weight = 0.0
        self.current_profit = 0.0
        self.current_step = 0
        self.n_penalty = 0
        self.n_bonuses = 0
        self.n_opportunities = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        i = self.current_step
        info = {
            "instance_id": self.instance_id,
            "step_idx": self.current_step,
            "action": action,
        }

        fits = self.weights[i] <= self.remaining_capacity
        terminated = False
        truncated = False
        reward = 0.0
        prev_profit = self.current_profit

        if action == 1:  # TAKE
            if fits:
                self.selected[i] = 1
                self.remaining_capacity -= self.weights[i]

                self.current_profit = float(
                    compute_profit(self.selected, self.profits, self.quad)
                )

                reward = (self.current_profit - prev_profit) / self.max_gain
            else:
                reward -= 0.1
                self.n_penalty += 1

        else:  # action == 0 (SKIP)
            if fits:
                reward -= 0 * self.profits[i] / self.max_gain
                self.n_opportunities += 1
            if not fits:
                reward += 0.1
                self.n_bonuses += 1
        self.current_step += 1

        terminated = self.current_step >= self.n or self.remaining_capacity <= 0
        # truncated = self.remaining_capacity <= 0 and not terminated

        info = {
            "instance_id": self.instance_id,
            "step_idx": i,  # decision index
            "action": action,
            "reward": reward,
            "lost_opportunities": self.n_opportunities,
            "penalties": self.n_penalty,
            "bonuses": self.n_bonuses,
            "remaining_capacity": self.remaining_capacity,
        }

        if terminated or truncated:
            info["final_profit"] = self.current_profit

        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _fractional_knapsack_upper_bound(self, start_idx):
        cap = self.remaining_capacity
        if cap <= 0 or start_idx >= self.n:
            return 0.0

        weights = self.weights[start_idx:]
        profits = self.profits[start_idx:]

        # Profit-to-weight ratio
        ratio = profits / weights
        order = np.argsort(ratio)[::-1]

        ub = 0.0
        remaining_cap = cap

        for idx in order:
            w = weights[idx]
            p = profits[idx]

            if w <= remaining_cap:
                ub += p
                remaining_cap -= w
            else:
                ub += p * (remaining_cap / w)
                break
        return ub

    def _get_obs(self):
        i = self.current_step
        if i >= self.n:
            return np.zeros(self.obs_dim, dtype=np.float32)
        # Standard
        w_i = self.weights[i] / self.max_w
        p_i = self.profits[i] / self.max_p
        fits = float(self.weights[i] <= self.remaining_capacity)
        remaining_cap = self.remaining_capacity / self.capacity

        # Remaining items
        remaining_weights = self.weights[i + 1 :]
        remaining_profits = self.profits[i + 1 :]

        # Linear quality
        pw_i = self.profits[i] / self.weights[i]
        remaining_pw = remaining_profits / remaining_weights
        pw_max = np.max(remaining_pw) if remaining_pw.size > 0 else pw_i
        pw_ratio_rel = pw_i / pw_max

        fk_ub = self._fractional_knapsack_upper_bound(i + 1)
        fk_ub_norm = fk_ub / self.max_linear

        # Quadratic interactions NOW
        quad_now = np.sum(self.quad[i, self.selected == 1])
        quad_now_norm = quad_now / (self.max_q * self.n)

        # Quadratic future of i : max quad of future item with i
        if i + 1 < self.n:
            quad_future_i_max = np.max(self.quad[i, i + 1 :]) / self.max_q
        else:
            quad_future_i_max = 0.0

        # Quadratic future of current set
        if np.any(self.selected) and i + 1 < self.n:
            quad_rem = self.quad[i + 1 :, self.selected == 1]
            max_quad_remaining = np.max(quad_rem) / self.max_q
        else:
            max_quad_remaining = 0.0

        # where are we? fraction selected and what else does fit?
        progress = i / self.n
        selected_frac = np.sum(self.selected) / self.n
        fraction_that_fit = (
            np.sum(remaining_weights <= self.remaining_capacity) / remaining_weights.size
            if remaining_weights.size > 0
            else 0.0
        )

        return np.array(
            [
                w_i,
                p_i,
                fits,
                remaining_cap,
                pw_ratio_rel,
                fk_ub_norm,
                quad_now_norm,
                quad_future_i_max,
                max_quad_remaining,
                progress,
                selected_frac,
                fraction_that_fit,
            ],
            dtype=np.float32,
        )


class DeepQAgent:
    """DQN agent for skip/take QKP environment."""

    def __init__(
        self,
        state_size,
        gamma=0.99,
        tau=0.005,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.925,
        lr=1e-4,
        batch_size=8,
        memory_size=50000,
        warmup_steps=0,
        build_model=True,
        model_name=None,
    ):
        self.state_size = state_size
        self.action_size = 2  # skip / take
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps

        self.memory = deque(maxlen=memory_size)

        if build_model:
            self.online_model = self._build_model(lr)
            self.target_model = self._build_model(lr)
            self.target_model.set_weights(self.online_model.get_weights())
        else:
            self.online_model = None
            self.target_model = None

        self.train_steps = 0
        self.episode_count = 0

        # logging replay batch
        self.losses = []
        self.steps = []
        self.qvals = []
        self.tds = []

        # logging per episode
        self.episode_rewards = []
        self.profits = []
        self.epsilons = []
        self.penalties = []
        self.bonuses = []
        self.remaining_caps = []
        self.episode_td_means = []
        self.episode_td_mins = []
        self.episode_td_maxs = []

        self.network_diffs = []

        log_dir = "logs/dqn_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = tf.summary.create_file_writer(log_dir)

        self.file_path_episode = "exc_2_results/train_episode_results.csv"
        self.file_path_steps = "exc_2_results/train_step_results.csv"
        os.makedirs("exc_2_results", exist_ok=True)
        file_exists = os.path.isfile(self.file_path_episode)

        with open(self.file_path_episode, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "episode",
                        "instance_id",
                        "epsilon",
                        "total_reward",
                        "td_diff_mean",
                        "td_diff_max",
                        "td_diff_min",
                        "network_diff",
                        "lost_opportunities",
                        "final_profit",
                        "n_penalties",
                        "n_bonuses",
                        "remaining_capacity",
                    ]
                )

        steps_exists = os.path.isfile(self.file_path_steps)

        with open(self.file_path_steps, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not steps_exists:
                writer.writerow(
                    [
                        "episode",
                        "step",
                        "instance_id",
                        "action",
                        "reward",
                        "lost_opportunities",
                        "penalties",
                        "bonuses",
                        "remaining_capacity",
                    ]
                )

    def _network_distance(self):
        online_weights = self.online_model.get_weights()
        target_weight = self.target_model.get_weights()
        return np.mean(
            [np.linalg.norm(ow - tw) for ow, tw in zip(online_weights, target_weight)]
        )

    # ------------------------
    # Model
    # ------------------------
    def _build_model(self, lr):
        model = models.Sequential(
            [
                layers.Input(shape=(self.state_size,)),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(128, activation="relu"),
                layers.Dense(self.action_size, activation="linear"),
            ]
        )
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
            loss=tf.keras.losses.Huber(delta=1.0),
        )
        return model

    def act_train(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)

        q_values = self.online_model.predict(state[np.newaxis], verbose=0)[0]
        return int(np.argmax(q_values))

    def act(self, state):
        q_values = self.online_model.predict(state[np.newaxis], verbose=0)[0]
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return None
        td_diffs = []
        batch = random.sample(self.memory, self.batch_size)

        states = np.array([b[0] for b in batch])
        next_states = np.array([b[3] for b in batch])

        q_current = self.online_model.predict(states, verbose=0)
        q_next_online = self.online_model.predict(next_states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)

        for i, (_, action, reward, _, done) in enumerate(batch):
            old_q = q_current[i, action]
            if done:
                target = reward
            else:
                a_online = np.argmax(q_next_online[i])
                target = reward + self.gamma * q_next_target[i, a_online]
            q_current[i, action] = target

            td_diff = target - old_q
            td_diffs.append(td_diff)

        history = self.online_model.fit(
            states,
            q_current,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0,
        )

        self.train_steps += 1
        self.soft_update_target_network()
        self.network_diffs.append(self._network_distance())

        self.losses.append(float(history.history["loss"][0]))
        self.tds.append(
            {
                "mean_diff": np.mean(np.abs(td_diffs)),
                "max_diff": np.max(np.abs(td_diffs)),
                "min_diff": np.min(np.abs(td_diffs)),
            }
        )
        self.qvals.append(
            {
                "mean_q": np.mean(q_current),
                "max_q": np.max(q_current),
                "min_q": np.min(q_current),
            }
        )

        with self.writer.as_default():
            tf.summary.scalar("train/loss", self.losses[-1], step=self.train_steps)
            tf.summary.scalar("train/reward", reward, step=self.train_steps)
            tf.summary.scalar(
                "train/td_mean", self.tds[-1]["mean_diff"], step=self.train_steps
            )
            tf.summary.scalar(
                "train/td_max", self.tds[-1]["max_diff"], step=self.train_steps
            )
            tf.summary.scalar(
                "train/q_mean", self.qvals[-1]["mean_q"], step=self.train_steps
            )
            tf.summary.scalar(
                "train/q_max", self.qvals[-1]["max_q"], step=self.train_steps
            )
            tf.summary.scalar(
                "train/network_diff", self.network_diffs[-1], step=self.train_steps
            )
        return self.losses[-1]

    def soft_update_target_network(self):
        online_weights = self.online_model.get_weights()
        target_weights = self.target_model.get_weights()

        new_weights = []
        for ow, tw in zip(online_weights, target_weights):
            new_weights.append(self.tau * ow + (1.0 - self.tau) * tw)

        self.target_model.set_weights(new_weights)

    def decay_epsilon(self, n_episodes):
        self.epsilon = max(
            self.epsilon_min,
            1.0 - self.episode_count * (1.0 - self.epsilon_min) / n_episodes,
        )

    def evaluate(self, env):
        state, _ = env.reset()

        while True:
            action = self.act(state)
            state, reward, terminated, _, info = env.step(action)
            if terminated:
                final_profit = info.get("final_profit", [0.0])
                return final_profit

    @classmethod
    def load_model(cls, model_path):
        model = models.load_model(model_path, compile=False)

        state_size = model.input_shape[-1]

        agent = cls(state_size=state_size, build_model=False, epsilon=0.0)

        agent.online_model = model

        agent.target_model = models.clone_model(model)
        agent.target_model.set_weights(model.get_weights())

        return agent

    def save_model(self, model_path):
        self.online_model.save(model_path)


def train_dqn(envs, agent_config, num_episodes=500, print_interval=50):
    assert len(envs) >= 2, "At least two environment are required."

    # Initialize agent
    example_env = envs[0]
    agent = DeepQAgent(
        state_size=example_env.observation_space.shape[0],
        **agent_config,
    )

    for ep in tqdm.tqdm(range(num_episodes)):
        env = random.choice(envs)
        state, _ = env.reset()
        total_reward = 0.0
        episode_rewards = []
        while True:
            action = agent.act_train(state)
            next_state, reward, terminated, _, info = env.step(action)

            agent.remember(state, action, reward, next_state, terminated)
            agent.replay()
            episode_rewards.append(reward)
            total_reward += reward
            state = next_state

            if ep % 25 == 0:
                with open(agent.file_path_steps, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            agent.episode_count,  # episode id
                            info["step_idx"],  # step id
                            info["instance_id"],
                            action,
                            reward,
                            info["lost_opportunities"],
                            info["penalties"],
                            info["bonuses"],
                            info["remaining_capacity"],
                        ]
                    )

            if terminated:
                agent.profits.append(info["final_profit"])
                agent.lost_opportunities = info.get("lost_opportunities", [0])
                agent.penalties.append(info.get("n_penalty", [0]))
                agent.bonuses.append(info.get("n_bonuses", [0]))
                agent.remaining_caps.append(info.get("remaining_capacity", [0]))
                break

        agent.decay_epsilon(n_episodes=num_episodes)
        agent.episode_count += 1
        agent.epsilons.append(agent.epsilon)
        agent.episode_rewards.append(episode_rewards)

        if (ep + 1) % print_interval == 0:
            print(
                f"Episode {ep + 1}/{num_episodes} - "
                f"Profit: {agent.profits[-1]:.4f}, "
                f"Epsilon: {agent.epsilons[-1]:.4f}"
            )

        # save model every 50 episodes
        if (ep + 1) % 500 == 0:
            name_file = f"exc_2_model/dqn_qkp_model_ep{ep + 1}.keras"
            agent.save_model(name_file)
            print(f"Saved model to {name_file}")

        last_td = (
            agent.tds[-1]
            if len(agent.tds) > 0
            else {"mean_diff": 0.0, "max_diff": 0.0, "min_diff": 0.0}
        )

        last_net_diff = agent.network_diffs[-1] if len(agent.network_diffs) > 0 else 0.0

        last_network_diff = (
            agent.network_diffs[-1] if len(agent.network_diffs) > 0 else 0.0
        )
        with open(agent.file_path_episode, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    agent.episode_count + 1,
                    info.get("instance_id", None),
                    agent.epsilon,
                    total_reward,
                    last_td["mean_diff"],
                    last_td["max_diff"],
                    last_td["min_diff"],
                    last_network_diff,
                    info.get("lost_opportunities", [0]),
                    info.get("final_profit", [0.0]),
                    info.get("penalties", [0]),
                    info.get("bonuses", [0]),
                    info.get("remaining_capacity", [0.0]),
                ]
            )
    agent.save_model(agent_config["model_name"])
    return agent


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model_dir = "exc_2_model"
    os.makedirs(model_dir, exist_ok=True)

    save = True
    num_episodes = 20
    print_interval = 5
    # agent_config = {
    #     "batch_size": 5,
    #     "gamma": 0.95,
    #     "tau": 0.01,
    #     "epsilon": 1.0,
    #     "epsilon_min": 0.05,
    #     "epsilon_decay": 0.999,
    #     "lr": 3e-4,
    #     "memory_size": 100_000,
    #     "warmup_steps": 1,
    #     "model_name": os.path.join(model_dir, "dqn_qkp_model.keras"),
    # }

    agent_config = {
        "batch_size": 96,
        "gamma": 0.95,
        "tau": 0.01,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.999,
        "lr": 3e-4,
        "memory_size": 100_000,
        "warmup_steps": 5000,
        "model_name": os.path.join(model_dir, "dqn_qkp_model.keras"),
    }

    # Load training instances
    folder = "InstancesEx2_train"
    instance_files = [
        os.path.join(folder, fname)
        for fname in os.listdir(folder)
        if fname.endswith(".txt")
    ]
    envs = []
    for i, f in enumerate(instance_files):
        n, cap, w, q = read_instance(f)
        p = [q[i][i] for i in range(n)]
        envs.append(DeepQEnv(w, p, q, cap, instance_id=i))

    # Train DQN agent
    trained_agent = train_dqn(envs, agent_config, num_episodes, print_interval)
