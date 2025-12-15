import datetime
import os
import random
from collections import deque


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tqdm

from e1_performance import read_instance
from e1_testing import compute_profit

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
        self.capacity = float(capacity)
        self.n = len(weights)
        self.instance_id = instance_id

        # Normalization constants
        self.max_w = float(np.max(self.weights))
        self.max_p = float(np.max(self.profits))
        self.max_q = float(np.max(np.triu(self.quad, k=1)))

        self.max_gain = float(np.max(self.profits + np.sum(self.quad, axis=1)))

        # Observation space size and action space
        self.obs_dim = 16
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self.selected = None
        self.remaining_capacity = None
        self.current_profit = 0.0
        self.current_idx = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.selected = np.zeros(self.n, dtype=np.int32)
        self.remaining_capacity = self.capacity
        self.current_weight = 0.0
        self.current_profit = 0.0
        self.current_idx = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        info = {"instance_id": self.instance_id}
        terminated = False
        truncated = False

        i = self.current_idx

        reward = 0.0
        prev_profit = self.current_profit

        if action == 1:  # TAKE
            if self.weights[i] <= self.remaining_capacity:
                self.selected[i] = 1
                self.remaining_capacity -= self.weights[i]

                self.current_profit = float(
                    compute_profit(self.selected, self.profits, self.quad)
                )
                reward = self.current_profit - prev_profit
                reward /= self.max_gain
            else:
                reward -= 0.1
        elif action == 0:
            if self.weights[i] <= self.remaining_capacity:
                # hypothetical_selected = self.selected.copy()
                # hypothetical_selected[i] = 1
                # hypothetical_profit = float(
                #     compute_profit(hypothetical_selected, self.profits, self.quad)
                # )
                # hypothetical_reward = hypothetical_profit - prev_profit
                # hypothetical_reward /= self.max_gain
                # reward = -hypothetical_reward
                pass
            else:
                reward += 0.1  # small bonus for skipping an item that doesn't fit (similar to penalty for trying to take it)

        self.current_idx += 1

        if self.current_idx == self.n - 1 or self.remaining_capacity <= 0:
            terminated = True
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
        i = self.current_idx
        # Item-specific
        w_i = self.weights[i] / self.max_w
        p_i = self.profits[i] / self.max_p
        fits = float(self.weights[i] <= self.remaining_capacity)
        pw_ratio = self.profits[i] / self.weights[i]
        pw_ratio_max = np.max(pw_ratio)
        pw_ratio = pw_ratio / pw_ratio_max

        # Quadratic contribution with already selected items
        if np.any(self.selected):
            quad_contrib = self.quad[i, self.selected == 1].sum() / (
                self.max_q * np.sum(self.selected)
            )
            best_synergy = np.max(self.quad[i, self.selected == 1])
        else:
            quad_contrib = 0.0
            best_synergy = 0.0

        selected_items = np.where(self.selected == 1)[0]
        remaining_cap = self.remaining_capacity / self.capacity
        progress = self.current_idx / self.n
        selected_frac = np.sum(self.selected) / self.n

        remaining_weights = self.weights[i + 1 :]
        remaining_profits = self.profits[i + 1 :]

        mean_remaining_weight = (
            np.mean(remaining_weights) / self.max_w if remaining_weights.size > 0 else 0.0
        )

        max_remaining_profit = (
            np.max(remaining_profits) / self.max_p if remaining_profits.size > 0 else 0.0
        )

        fraction_that_fit = (
            np.sum(remaining_weights <= self.remaining_capacity) / self.n
            if remaining_weights.size > 0
            else 0.0
        )

        if selected_items.size > 0 and i + 1 < self.n:
            quad_remaining = self.quad[
                i + 1 :, selected_items
            ]  # shape: (remaining_items, selected)

            max_quad_remaining = np.max(quad_remaining) / self.max_q
            mean_quad_remaining = np.mean(quad_remaining) / self.max_q
        else:
            max_quad_remaining = 0.0
            mean_quad_remaining = 0.0

        fk_ub = self._fractional_knapsack_upper_bound(i + 1)
        fk_ub_norm = fk_ub / (self.max_q * self.n)

        obs = [
            w_i,
            p_i,
            fits,
            pw_ratio,
            pw_ratio_max,
            quad_contrib,
            best_synergy,
            remaining_cap,
            progress,
            selected_frac,
            mean_remaining_weight,
            max_remaining_profit,
            fraction_that_fit,
            max_quad_remaining,
            mean_quad_remaining,
            fk_ub_norm,
        ]

        return np.array(obs, dtype=np.float32)


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

        log_dir = "logs/dqn_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = tf.summary.create_file_writer(log_dir)

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
        return self.losses[-1]

    def soft_update_target_network(self):
        online_weights = self.online_model.get_weights()
        target_weights = self.target_model.get_weights()

        new_weights = []
        for ow, tw in zip(online_weights, target_weights):
            new_weights.append(self.tau * ow + (1.0 - self.tau) * tw)

        self.target_model.set_weights(new_weights)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def evaluate(self, env):
        state, _ = env.reset()

        while True:
            action = self.act(state)
            state, reward, terminated, _, info = env.step(action)
            if terminated:
                final_profit = info.get("final_profit", 0.0)
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

            if terminated:
                agent.profits.append(info["final_profit"])
                break

        agent.decay_epsilon()
        agent.episode_count += 1
        agent.epsilons.append(agent.epsilon)
        agent.episode_rewards.append(episode_rewards)

        if (ep + 1) % print_interval == 0:
            print(
                f"Episode {ep + 1}/{num_episodes} - "
                f"Profit: {agent.profits[-1]:.4f}, "
                f"Epsilon: {agent.epsilons[-1]:.4f}"
            )

        with agent.writer.as_default():
            tf.summary.scalar(
                "episode/profit",
                agent.profits[-1],
                step=agent.episode_count,
            )
            tf.summary.scalar(
                "episode/epsilon",
                agent.epsilons[-1],
                step=agent.episode_count,
            )

    return agent


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    save = True
    num_episodes = 5000
    print_interval = 100
    agent_config = {
        "batch_size": 64,
        "gamma": 0.95,
        "tau": 0.005,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.999,
        "lr": 3e-4,
        "memory_size": 100000,
        "warmup_steps": 1000,
    }

    # Load training instances
    folder = "InstancesEx2_train"
    instance_files = [
        os.path.join(folder, fname)
        for fname in os.listdir(folder)
        if fname.endswith(".txt")
    ]
    envs = []
    for f in instance_files:
        n, cap, w, q = read_instance(f)
        p = [q[i][i] for i in range(n)]
        envs.append(DeepQEnv(w, p, q, cap, instance_id=0))

    # Train DQN agent
    trained_agent = train_dqn(envs, agent_config, num_episodes, print_interval)

    if save:
        # Save the trained model
        model_dir = "exc_2_model"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"exc_2_model.keras")
        trained_agent.online_model.save(model_path)

        print(f"Trained model saved to {model_path}")
