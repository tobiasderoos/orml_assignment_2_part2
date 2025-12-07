import datetime
import os
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tqdm

from e1_greedy import read_instance, compute_profit


def greedy_heuristic(weights, profits, quad, capacity):
    n = len(weights)
    remaining_capacity = capacity
    total_profit = 0
    marginal_profits = []
    selected = set()

    while remaining_capacity > 0:
        candidates = [i for i in range(n) if weights[i] <= remaining_capacity]
        candidates = [
            i
            for i in range(n)
            if i not in selected and weights[i] <= remaining_capacity
        ]

        if not candidates:
            break
        best_item = max(candidates, key=lambda i: profits[i] / weights[i])
        selected.add(best_item)

        marginal_profit = profits[best_item] + np.sum(
            quad[best_item, list(selected)]
        )
        marginal_profits.append(marginal_profit)

        total_profit += marginal_profit

        remaining_capacity -= weights[best_item]

    return total_profit, marginal_profits


class PERBuffer:
    """
    Prioritized Experience Replay buffer using proportional prioritization.

    p_i = (|TD error| + eps)^alpha
    P(i) = p_i / sum_j p_j

    Experiences with high TD error are sampled more often.
    """

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # how much prioritization to use
        self.buffer = []
        self.priorities = []
        self.pos = 0  # cyclic position for overwriting

        # Small constant to avoid zero probability
        self.eps = 1e-6

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, td_error=None):
        """Add experience with priority = td_error or max priority."""

        if td_error is None:
            # max priority ensures new samples are sampled at least once
            priority = max(self.priorities, default=1.0)
        else:
            priority = abs(td_error) + self.eps

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            # Overwrite oldest entry
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch with importance sampling weights.

        Returns:
            batch, indices, weights
        """
        if len(self.buffer) == 0:
            raise ValueError("Sampling from empty PER buffer")

        # Compute probabilities
        scaled_priorities = np.array(self.priorities) ** self.alpha
        probs = scaled_priorities / np.sum(scaled_priorities)

        # Sample indices according to probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        batch = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize to 1.0

        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        """Update priorities after replay step."""
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td) + self.eps


class QEnv(gym.Env):
    """
    Gymnasium environment for the QPS with improved reward normalization.
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

        # Normalization factors
        self.max_w = np.max(self.weights)
        self.max_p = np.max(self.profits)
        self.max_q = np.max(np.triu(self.quad, k=1))

        self.max_gain = np.max(self.profits + np.sum(self.quad, axis=1))

        # Observation space consists of:
        # - selected items (n)
        # - available items (n)
        # - weights (n)
        # - profits (n)
        # - quadratic profits (n)
        # and two scalers for remaining capacity and current weight
        obs_size = 11 * self.n + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n)

        # Internal state
        self.selected = None
        self.available = None
        self.remaining_capacity = None
        self.current_weight = 0
        self.current_reward = 0

        # Compute greedy profit
        self.greedy_profit, self._marginal_profits = greedy_heuristic(
            self.weights, self.profits, self.quad, self.capacity
        )

    def reset(self, *, seed=None):
        super().reset(seed=seed)

        self.selected = np.zeros(self.n, dtype=np.int32)
        self.available = np.ones(self.n, dtype=np.int32)
        self.remaining_capacity = self.capacity
        self.current_weight = 0.0
        self.current_profit = 0.0

        return self._get_obs(), {}

    def step(self, action, steps):
        info = {"instance_id": self.instance_id}
        terminated = False
        truncated = False

        if (
            self.selected[action] == 1
            or self.weights[action] > self.remaining_capacity
        ):
            info["warning"] = "invalid_action_reached"
            print("Warning: Invalid action.")
            raise ValueError("Invalid action taken in QEnv.")

        # Calculate marginal profit
        marginal = self._marginal_profit(action)
        greedy_marginal = (
            self._marginal_profits[steps]
            if steps < len(self._marginal_profits)
            else 1.0
        )
        eps = 1e-6
        ratio = (marginal - greedy_marginal) / (abs(greedy_marginal) + eps)
        ratio = np.clip(ratio, -1.0, +1.0)
        reward = ratio
        # Apply selection
        self.selected[action] = 1
        self.available[action] = 0
        self.remaining_capacity -= self.weights[action]
        self.current_weight += self.weights[action]
        self.current_reward += reward

        # Check termination
        if not np.any(self.get_feasible_mask()):
            terminated = True
            final_profit = compute_profit(
                self.selected, self.profits, self.quad
            )
            terminal_reward = (final_profit / self.greedy_profit) - 1
            terminal_reward *= 10
            terminal_reward = np.clip(terminal_reward, -10, +10)

            info["reason"] = "no_feasible_items"
            info["optimality_ratio"] = terminal_reward
            info["reward ratios"] = reward / terminal_reward
            reward += terminal_reward
        return self._get_obs(), reward, terminated, truncated, info

    def get_feasible_mask(self):
        return (self.available == 1) & (self.weights <= self.remaining_capacity)

    def _marginal_profit(self, i):
        selected_items = np.where(self.selected == 1)[0]
        lin_profit = self.profits[i]
        if len(selected_items) == 0:
            return lin_profit
        return lin_profit + np.sum(self.quad[i, selected_items])

    def _get_obs(self):
        fits = (self.weights <= self.remaining_capacity).astype(np.float32)

        norm_w = (self.weights / self.max_w).astype(np.float32)
        norm_p = (self.profits / self.max_p).astype(np.float32)

        pw_ratio = self.profits / self.weights
        pw_ratio = (pw_ratio / pw_ratio.max()).astype(np.float32)

        self.available = ((self.selected == 0) & (fits == 1)).astype(np.float32)

        selected_items = np.where(self.selected == 1)[0]
        if len(selected_items) == 0:
            marginal_profit_vector = self.profits / self.max_p
        else:
            marginal_profit_vector = self.profits / self.max_p + (
                self.quad[:, selected_items].sum(axis=1) / self.max_q
            )
            marginal_profit_vector[selected_items] = 0.0  # to be sure
            marginal_profit_vector = marginal_profit_vector.astype(np.float32)
        # average quadratic profit of available items
        quad_no_diag = self.quad.copy()
        np.fill_diagonal(quad_no_diag, 0)
        avg_quad_int = quad_no_diag.mean(axis=1)
        avg_quad_int = avg_quad_int / self.max_q

        def rank_normalized(x):
            return np.argsort(np.argsort(x)) / (len(x) - 1)

        profit_rank = rank_normalized(self.profits)
        weight_rank = rank_normalized(self.weights)
        pw_rank = rank_normalized(self.profits / self.weights)
        avg_quad_int_rank = rank_normalized(avg_quad_int)

        remaining_cap = self.remaining_capacity / self.capacity
        items_left = np.sum(self.available) / self.n

        scalars = np.array(
            [remaining_cap, items_left],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                self.available,  # n
                fits.astype(np.float32),  # n
                norm_w,  # n
                norm_p,  # n
                pw_ratio,  # n
                marginal_profit_vector,  # n
                avg_quad_int,  # n
                profit_rank,  # n
                weight_rank,  # n
                pw_rank,  # n
                avg_quad_int_rank,  # n
                scalars,  # 2 scalars
            ]
        )


class QAgent:
    """Simple DQN agent for QPS environment."""

    def __init__(
        self,
        state_size,
        action_size,
        gamma=0.9975,
        tau=0.001,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999,
        lr=0.0002,
        batch_size=128,
        memory_size=25000,
        target_update_interval=250,
        warmup_steps=100,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.target_update_interval = target_update_interval
        self.warmup_steps = warmup_steps
        # self.memory = PERBuffer(memory_size)
        self.memory = deque(maxlen=memory_size)
        # Build networks
        self.online_model = self._build_model(lr)
        self.target_model = self._build_model(lr)
        self.target_model.set_weights(self.online_model.get_weights())

        # Counters
        self.train_steps = 0
        self.episode_count = 0

        # metrics tracking
        self.losses = []
        self.recent_profits = deque(maxlen=100)
        self.profits = []
        self.recent_optimality = deque(maxlen=100)
        self.recent_rewards = deque(maxlen=100)
        self.recent_episode_lengths = deque(maxlen=100)
        self.ratio_rewards = []
        # Per-episode tracking
        self.all_episode_rewards = []
        self.all_episode_profits = []
        self.all_episode_optimality = []

        # Q-value statistics
        self.q_value_history = deque(maxlen=1000)
        self.q_value_std_history = deque(maxlen=1000)
        self.td_error_history = deque(maxlen=1000)

        # Action statistics
        self.action_entropy_history = deque(maxlen=100)

        # Gradient/training stats
        self.grad_norms = deque(maxlen=100)

        # Instance performance tracking
        self.instance_performance = {}

        log_dir = "logs/simple_dqn_" + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
        self.writer = tf.summary.create_file_writer(log_dir)

    def _build_model(self, lr):
        """Simple 2-layer network"""
        model = models.Sequential(
            [
                layers.Input(shape=(self.state_size,)),
                layers.Dense(256, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(self.action_size, activation="linear"),
            ]
        )
        model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss="mse")
        return model

    def act(self, state, feasible_mask):
        """Select action using epsilon-greedy"""
        feasible_actions = np.where(feasible_mask)[0]
        if len(feasible_actions) == 0:
            return 0

        # Exploration
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(feasible_actions))

        # Exploitation
        q_values = self.online_model.predict(state[np.newaxis, :], verbose=0)[0]

        q_values[~feasible_mask] = -np.inf
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done, next_feasible):
        """Store experience"""
        transition = (state, action, reward, next_state, done, next_feasible)
        # self.memory.add(transition)
        self.memory.append(transition)

    def replay(self):
        """One gradient step from replay buffer."""
        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return None

        # batch, indices, weights = self.memory.sample(self.batch_size, beta=0.4)

        batch = random.sample(self.memory, self.batch_size)

        # (state, action, reward, next_state, done, next_feasible)
        states = np.array([t[0] for t in batch])
        next_states = np.array([t[3] for t in batch])

        # Q(s',·) from online and target networks
        q_current = self.online_model.predict(states, verbose=0)

        q_vals = q_current.flatten()
        self.q_value_history.append(
            {
                "mean": float(np.mean(q_vals)),
                "max": float(np.max(q_vals)),
                "min": float(np.min(q_vals)),
                "std": float(np.std(q_vals)),
            }
        )

        q_next_online = self.online_model.predict(next_states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)

        q_original = q_current.copy()
        td_errors = []
        for idx, (s, a, r, s2, done, feas2) in enumerate(batch):
            old_q = q_original[idx, a]
            if done:
                target = r
            else:
                # 1) online network selects best next action
                q_online_masked = q_next_online[idx].copy()
                if feas2 is not None:
                    q_online_masked[~feas2] = -np.inf
                a_online = np.argmax(q_online_masked)

                # 2) target network evaluates that action
                target = r + self.gamma * q_next_target[idx, a_online]
            td_errors.append(abs(old_q - target))
            q_current[idx, a] = target
        # self.memory.update_priorities(indices, td_errors)

        history = self.online_model.fit(
            states,
            q_current,
            epochs=1,
            verbose=0,
            batch_size=self.batch_size,
        )
        loss = float(history.history["loss"][0])
        self.losses.append(loss)
        # One gradient step of batch is done.
        self.train_steps += 1

        self.soft_update_target_network()
        # if self.train_steps % self.target_update_interval == 0:
        #     self.update_target_network()
        return (loss, td_errors)

    def update_target_network(self):
        """Hard update of target network"""
        self.target_model.set_weights(self.online_model.get_weights())

    def soft_update_target_network(self):
        online_weights = self.online_model.get_weights()
        target_weights = self.target_model.get_weights()

        new_weights = []
        for ow, tw in zip(online_weights, target_weights):
            nw = self.tau * ow + (1 - self.tau) * tw
            new_weights.append(nw)

        self.target_model.set_weights(new_weights)

    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def log_diagnostics(self):
        """Log diagnostics to TensorBoard."""

        step = self.episode_count

        if step is None:
            return

        with self.writer.as_default():
            if self.losses:
                tf.summary.scalar("Training/Loss", self.losses[-1], step=step)

            if self.recent_rewards:
                tf.summary.scalar(
                    "Episode/Reward", list(self.recent_rewards)[-1], step=step
                )

            tf.summary.scalar("Training/Epsilon", self.epsilon, step=step)

            if self.recent_optimality:
                tf.summary.scalar(
                    "Episode/Optimality",
                    list(self.recent_optimality)[-1],
                    step=step,
                )

            if self.q_value_history:
                recent_q = list(self.q_value_history)[-1]  # last entry
                tf.summary.scalar("Q/Mean", recent_q["mean"], step=step)
                tf.summary.scalar("Q/Max", recent_q["max"], step=step)
                tf.summary.scalar("Q/Min", recent_q["min"], step=step)
                tf.summary.scalar(
                    "Q/Range", recent_q["max"] - recent_q["min"], step=step
                )

            if self.td_error_history:
                tf.summary.scalar(
                    "TD_Error/Mean", list(self.td_error_history)[-1], step=step
                )

            if self.ratio_rewards:
                tf.summary.scalar(
                    "Episode/RewardRatio",
                    list(self.ratio_rewards)[-1],
                    step=step,
                )

            # # === GRADIENT NORM ===
            # if self.gradient_norms:
            #     tf.summary.scalar(
            #         "Gradients/Norm",
            #         np.mean(list(self.gradient_norms)[-10:]),
            #         step=step,
            #     )

            # # === TARGET NETWORK DIVERGENCE ===
            # div = self.compute_target_divergence()
            # tf.summary.scalar("Target/Divergence", div, step=step)

            # # === MEMORY SIZE ===
            # tf.summary.scalar("Memory/Size", len(self.memory), step=step)

    def _get_metrics(self):
        """Return dictionary of metrics"""
        metrics = {
            "avg_reward_50": np.mean(list(self.recent_rewards)[-50:])
            if self.recent_rewards
            else 0,
            "avg_reward_ratio_50": np.mean(list(self.ratio_rewards)[-50:])
            if self.ratio_rewards
            else 0,
            "avg_profit_50": np.mean(list(self.recent_profits)[-50:])
            if self.recent_profits
            else 0,
            "max_profit": max(self.recent_profits)
            if self.recent_profits
            else 0,
            "avg_optimality_50": np.mean(list(self.recent_optimality)[-50:])
            if self.recent_optimality
            else 0,
            "avg_length_50": np.mean(list(self.recent_episode_lengths)[-50:])
            if self.recent_episode_lengths
            else 0,
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
            "memory_size": len(self.memory),
            "avg_loss_50": np.mean(list(self.losses)[-50:])
            if self.losses
            else 0,
            "avg_q_value": np.mean(self.q_value_history)
            if self.q_value_history
            else 0,
            "q_value_std": np.mean(self.q_value_std_history)
            if self.q_value_std_history
            else 0,
            "action_diversity": np.mean(self.action_entropy_history)
            if self.action_entropy_history
            else 0,
        }

        return metrics

    def save(self, path):
        """Save only the online model"""
        self.online_model.save(path)

    def load(self, path):
        self.online_model = models.load_model(path)
        self.target_model.set_weights(self.online_model.get_weights())


def train_episode(env, agent):
    """Train one episode"""
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    episode_q_values = []
    episode_td_errors = []
    while True:
        # Get feasible actions
        feasible = env.get_feasible_mask()
        if not np.any(feasible):
            break

        # Select and perform action
        action = agent.act(state, feasible)
        next_state, reward, terminated, _, info = env.step(action, steps)

        # Store experience
        next_feasible = env.get_feasible_mask() if not terminated else None
        agent.remember(
            state, action, reward, next_state, terminated, next_feasible
        )

        replay_out = agent.replay()
        if replay_out is not None:
            loss, td_errors = replay_out
            episode_td_errors.extend(td_errors)

        loss, td_errors = replay_out if replay_out is not None else (0, [])
        total_reward += reward

        steps += 1

        if terminated:
            final_profit = compute_profit(env.selected, env.profits, env.quad)
            optimality_ratio = info.get("optimality_ratio", 0)

            agent.recent_rewards.append(total_reward)
            agent.recent_profits.append(final_profit)
            agent.recent_optimality.append(optimality_ratio)
            agent.recent_episode_lengths.append(steps)

            agent.all_episode_rewards.append(total_reward)
            agent.all_episode_profits.append(final_profit)
            agent.all_episode_optimality.append(optimality_ratio)

            instance_id = info.get("instance_id", 0)
            if instance_id not in agent.instance_performance:
                agent.instance_performance[instance_id] = []
            agent.instance_performance[instance_id].append(final_profit)

            if len(episode_q_values) > 0:
                avg_q_std = np.std(episode_q_values)
                agent.action_entropy_history.append(avg_q_std)

            break

        state = next_state

    # Decay exploration
    agent.decay_epsilon()
    agent.log_diagnostics()
    agent.episode_count += 1
    return episode_td_errors if episode_td_errors else None


def train(
    instance_folder,
    num_episodes=1000,
    save_path="exc2.keras",
    max_instances=None,
    print_interval=10,
):
    """Main training loop"""

    # Load instances
    instance_files = sorted(
        [
            os.path.join(instance_folder, f)
            for f in os.listdir(instance_folder)
            if os.path.isfile(os.path.join(instance_folder, f))
        ]
    )

    if max_instances:
        instance_files = instance_files[:max_instances]

    print(f"Loading {len(instance_files)} instances.")
    envs = []
    for idx, fpath in enumerate(instance_files):
        n, cap, w, q = read_instance(fpath)
        p = [q[i][i] for i in range(n)]
        env = QEnv(w, p, q, cap, instance_id=idx)
        envs.append(env)
        print(
            f"  Instance {idx}: n={n}, capacity={cap}, greedy_profit={env.greedy_profit:.2f}"
        )
    example_env = envs[0]
    # Create agent
    agent = QAgent(
        state_size=example_env.observation_space.shape[0],
        action_size=example_env.action_space.n,
        batch_size=64,
    )
    print(f"\nTraining for {num_episodes} episodes.")
    print("\n" + "-" * 80)

    # Training loop
    for episode in tqdm.tqdm(range(num_episodes)):
        # Pick random instance
        env = random.choice(envs)

        # Train one episode
        td_errors = train_episode(env, agent)
        profit = agent.all_episode_profits[-1]
        opt = agent.all_episode_optimality[-1]
        reward = agent.all_episode_rewards[-1]
        avg_td_error = np.mean(td_errors) if td_errors is not None else 0

        def _get_avg_last(data, n=50):
            return np.mean(list(data)[-n:]) if data else 0

        # Print progress every 10 episodes
        if episode % print_interval == 0:
            print(
                f"Episode {episode:4d} | "
                f"Profit: {profit:6.0f}  (avg50: {_get_avg_last(agent.all_episode_profits):6.0f}) | "
                f"Opt: {opt:5.2f}  (avg50: {_get_avg_last(agent.all_episode_optimality):5.2f}) | "
                f"TD Error: {avg_td_error:6.3f} | "
                f"Train Steps: {agent.train_steps:6d} | "
                f"Reward: {reward:6.1f}  (avg50: {_get_avg_last(agent.all_episode_rewards):6.1f}) | "
                f"Eps: {agent.epsilon:.3f}"
            )
        # Save checkpoint every 50 episodes
        if (episode + 1) % 50 == 0:
            agent.save(save_path)
            print(f"\n  → Checkpoint saved to {save_path}")

    # Final save
    agent.save(save_path)
    print("\n" + "-" * 80)
    print(f"Training complete: model saved to {save_path}")
    return agent


def smooth(data, window=20):
    """Apply smoothing to noisy data."""
    if len(data) < window:
        return np.array(data)
    return uniform_filter1d(
        np.array(data, dtype=float), size=window, mode="nearest"
    )


def plot_rl_diagnostics(agent, save_path=None):
    """
    Comprehensive RL training diagnostics.

    Figures:
    1. Optimality (profit relative to greedy)
    2. Learning progress (rewards)
    3. Stability (loss and TD errors)
    4. Q-value health
    5. Exploration (epsilon)
    6. Episode statistics
    """

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("DQN Training Diagnostics", fontsize=14, fontweight="bold")

    episodes = np.arange(len(agent.all_episode_rewards))

    # =========================================
    # 1. OPTIMALITY RATIO (Most Important!)
    # =========================================
    ax1 = axes[0, 0]
    opt = np.array(agent.all_episode_optimality)
    ax1.plot(episodes, opt, alpha=0.3, color="blue", label="Raw")
    ax1.plot(smooth(opt, 50), color="blue", linewidth=2, label="Smoothed (50)")
    ax1.axhline(
        y=1.0, color="red", linestyle="--", linewidth=2, label="Greedy baseline"
    )
    ax1.fill_between(
        episodes,
        1.0,
        opt,
        where=(opt > 1.0),
        alpha=0.3,
        color="green",
        label="Beating greedy",
    )
    ax1.fill_between(
        episodes,
        opt,
        1.0,
        where=(opt < 1.0),
        alpha=0.3,
        color="red",
        label="Below greedy",
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Profit / Greedy Profit")
    ax1.set_title("Optimality Ratio (>1 = beating greedy)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    final_100_opt = np.mean(opt[-100:]) if len(opt) >= 100 else np.mean(opt)
    best_opt = np.max(opt) if len(opt) > 0 else 0
    ax1.text(
        0.02,
        0.98,
        f"Final 100 avg: {final_100_opt:.3f}\nBest: {best_opt:.3f}",
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # =========================================
    # 2. LEARNING PROGRESS (Rewards)
    # =========================================
    ax2 = axes[0, 1]
    rewards = np.array(agent.all_episode_rewards)
    ax2.plot(episodes, rewards, alpha=0.3, color="green", label="Raw")
    ax2.plot(
        smooth(rewards, 50), color="green", linewidth=2, label="Smoothed (50)"
    )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Reward")
    ax2.set_title("Learning Progress (Episode Rewards)")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Trend line
    if len(rewards) > 100:
        z = np.polyfit(episodes, rewards, 1)
        p = np.poly1d(z)
        ax2.plot(
            episodes,
            p(episodes),
            "r--",
            alpha=0.5,
            label=f"Trend (slope={z[0]:.4f})",
        )
        ax2.legend(loc="lower right", fontsize=8)

    # =========================================
    # 3. STABILITY (Loss)
    # =========================================
    ax3 = axes[1, 0]
    if agent.losses:
        losses = np.array(agent.losses)
        steps = np.arange(len(losses))
        ax3.plot(steps, losses, alpha=0.3, color="orange", label="Raw")
        ax3.plot(
            smooth(losses, 100),
            color="orange",
            linewidth=2,
            label="Smoothed (100)",
        )
        ax3.set_xlabel("Training Step")
        ax3.set_ylabel("Loss")
        ax3.set_title("Training Stability (Loss)")
        ax3.set_yscale("log")  # Log scale for loss
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Check for instability
        if len(losses) > 200:
            recent_std = np.std(losses[-200:])
            early_std = np.std(losses[:200])
            stability_ratio = recent_std / (early_std + 1e-8)
            stability_status = "Stable" if stability_ratio < 2 else "Unstable!"
            ax3.text(
                0.02,
                0.98,
                f"Stability: {stability_status}\nRecent σ / Early σ: {stability_ratio:.2f}",
                transform=ax3.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
    else:
        ax3.text(
            0.5,
            0.5,
            "No loss data",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )

    # =========================================
    # 4. Q-VALUE HEALTH
    # =========================================
    ax4 = axes[1, 1]
    if agent.q_value_history:
        q_data = list(agent.q_value_history)
        q_means = [d["mean"] for d in q_data]
        q_maxs = [d["max"] for d in q_data]
        q_mins = [d["min"] for d in q_data]
        q_stds = [d["std"] for d in q_data]

        steps = np.arange(len(q_means))
        ax4.plot(steps, q_means, label="Mean Q", color="blue", linewidth=1.5)
        ax4.fill_between(
            steps,
            q_mins,
            q_maxs,
            alpha=0.2,
            color="blue",
            label="Min-Max range",
        )
        ax4.plot(
            steps,
            q_stds,
            label="Std Q",
            color="red",
            linewidth=1,
            linestyle="--",
        )

        ax4.set_xlabel("Training Step")
        ax4.set_ylabel("Q-Value")
        ax4.set_title("Q-Value Health (check for divergence/collapse)")
        ax4.legend(loc="upper left", fontsize=8)
        ax4.grid(True, alpha=0.3)

        # Detect issues
        issues = []
        if len(q_means) > 100:
            if np.max(q_means[-100:]) > 1000:
                issues.append("⚠️ Q-values exploding!")
            if np.mean(q_stds[-100:]) < 0.01:
                issues.append("⚠️ Q-values collapsed!")
            if np.std(q_means[-100:]) > np.mean(np.abs(q_means[-100:])):
                issues.append("⚠️ High Q variance!")

        if issues:
            ax4.text(
                0.02,
                0.98,
                "\n".join(issues),
                transform=ax4.transAxes,
                fontsize=9,
                verticalalignment="top",
                color="red",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            )
    else:
        ax4.text(
            0.5,
            0.5,
            "No Q-value data",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )

    # =========================================
    # 5. EXPLORATION (Epsilon)
    # =========================================
    ax5 = axes[2, 0]
    # Reconstruct epsilon history
    eps_history = []
    eps = 1.0
    eps_min = agent.epsilon_min
    eps_decay = agent.epsilon_decay
    for _ in range(len(episodes)):
        eps_history.append(eps)
        eps = max(eps_min, eps * eps_decay)

    ax5.plot(episodes, eps_history, color="purple", linewidth=2)
    ax5.axhline(
        y=eps_min, color="red", linestyle="--", label=f"ε_min = {eps_min}"
    )
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Epsilon")
    ax5.set_title("Exploration Rate (Epsilon Decay)")
    ax5.legend(loc="upper right", fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Mark current epsilon
    ax5.scatter(
        [len(episodes) - 1],
        [agent.epsilon],
        color="red",
        s=100,
        zorder=5,
        label=f"Current: {agent.epsilon:.3f}",
    )

    # =========================================
    # 6. EPISODE LENGTH / ITEMS SELECTED
    # =========================================
    ax6 = axes[2, 1]
    if agent.recent_episode_lengths:
        lengths = list(agent.recent_episode_lengths)
        # Extend to full history if available
        if hasattr(agent, "all_episode_lengths"):
            lengths = agent.all_episode_lengths

        ep_range = np.arange(len(lengths))
        ax6.plot(ep_range, lengths, alpha=0.3, color="teal", label="Raw")
        ax6.plot(
            smooth(lengths, 20),
            color="teal",
            linewidth=2,
            label="Smoothed (20)",
        )
        ax6.set_xlabel("Episode")
        ax6.set_ylabel("Items Selected")
        ax6.set_title("Episode Length (Items Selected per Episode)")
        ax6.legend(loc="lower right", fontsize=8)
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(
            0.5,
            0.5,
            "No episode length data",
            ha="center",
            va="center",
            transform=ax6.transAxes,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_profit_analysis(agent, save_path=None):
    """
    Detailed profit analysis figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Profit Analysis", fontsize=14, fontweight="bold")

    profits = np.array(agent.all_episode_profits)
    opt = np.array(agent.all_episode_optimality)
    episodes = np.arange(len(profits))

    # =========================================
    # 1. Raw Profits
    # =========================================
    ax1 = axes[0, 0]
    ax1.plot(episodes, profits, alpha=0.3, color="green")
    ax1.plot(smooth(profits, 50), color="green", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Profit")
    ax1.set_title("Absolute Profit per Episode")
    ax1.grid(True, alpha=0.3)

    # =========================================
    # 2. Profit Distribution (Early vs Late)
    # =========================================
    ax2 = axes[0, 1]
    split = len(profits) // 2
    if split > 0:
        early = profits[:split]
        late = profits[split:]
        ax2.hist(
            early,
            bins=30,
            alpha=0.5,
            label=f"First half (μ={np.mean(early):.1f})",
            color="red",
        )
        ax2.hist(
            late,
            bins=30,
            alpha=0.5,
            label=f"Second half (μ={np.mean(late):.1f})",
            color="green",
        )
        ax2.set_xlabel("Profit")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Profit Distribution: Early vs Late Training")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # =========================================
    # 3. Optimality Over Time (Rolling Stats)
    # =========================================
    ax3 = axes[1, 0]
    window = 100
    if len(opt) >= window:
        rolling_mean = np.array(
            [np.mean(opt[max(0, i - window) : i + 1]) for i in range(len(opt))]
        )
        rolling_std = np.array(
            [np.std(opt[max(0, i - window) : i + 1]) for i in range(len(opt))]
        )

        ax3.plot(
            episodes,
            rolling_mean,
            color="blue",
            linewidth=2,
            label="Rolling mean (100)",
        )
        ax3.fill_between(
            episodes,
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.2,
            color="blue",
            label="±1 std",
        )
        ax3.axhline(
            y=1.0,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Greedy baseline",
        )
    else:
        ax3.plot(episodes, opt, color="blue")
        ax3.axhline(y=1.0, color="red", linestyle="--")

    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Optimality Ratio")
    ax3.set_title("Optimality with Confidence Band")
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # =========================================
    # 4. Performance Improvement Rate
    # =========================================
    ax4 = axes[1, 1]
    if len(opt) >= 200:
        # Compute improvement in chunks
        chunk_size = len(opt) // 10
        chunk_means = [
            np.mean(opt[i * chunk_size : (i + 1) * chunk_size])
            for i in range(10)
        ]
        chunk_labels = [f"{i * 10}-{(i + 1) * 10}%" for i in range(10)]

        colors = ["red" if m < 1.0 else "green" for m in chunk_means]
        bars = ax4.bar(
            range(10), chunk_means, color=colors, alpha=0.7, edgecolor="black"
        )
        ax4.axhline(y=1.0, color="red", linestyle="--", linewidth=2)
        ax4.set_xticks(range(10))
        ax4.set_xticklabels(chunk_labels, rotation=45)
        ax4.set_xlabel("Training Progress")
        ax4.set_ylabel("Avg Optimality")
        ax4.set_title("Performance by Training Stage")
        ax4.grid(True, alpha=0.3, axis="y")

        # Improvement from first to last chunk
        improvement = chunk_means[-1] - chunk_means[0]
        ax4.text(
            0.02,
            0.98,
            f"Improvement: {improvement:+.3f}",
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    else:
        ax4.text(
            0.5,
            0.5,
            "Need more episodes for this plot",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_training_summary(agent, save_path=None):
    """
    Single summary figure for quick assessment.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training Summary", fontsize=14, fontweight="bold")

    opt = np.array(agent.all_episode_optimality)
    rewards = np.array(agent.all_episode_rewards)
    episodes = np.arange(len(opt))

    # 1. Optimality
    ax1 = axes[0]
    ax1.plot(episodes, opt, alpha=0.2, color="blue")
    ax1.plot(smooth(opt, 50), color="blue", linewidth=2)
    ax1.axhline(y=1.0, color="red", linestyle="--", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Optimality")
    ax1.set_title("Profit / Greedy")
    ax1.grid(True, alpha=0.3)

    # 2. Rewards
    ax2 = axes[1]
    ax2.plot(episodes, rewards, alpha=0.2, color="green")
    ax2.plot(smooth(rewards, 50), color="green", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.set_title("Episode Reward")
    ax2.grid(True, alpha=0.3)

    # 3. Key Metrics Box
    ax3 = axes[2]
    ax3.axis("off")

    metrics_text = f"""
    TRAINING METRICS
    ────────────────────
    Episodes:        {len(opt)}
    Training Steps:  {agent.train_steps}
    
    OPTIMALITY
    ────────────────────
    Final (last 100):  {np.mean(opt[-100:]):.4f}
    Best:              {np.max(opt):.4f}
    % > greedy:        {100 * np.mean(opt > 1.0):.1f}%
    
    REWARDS
    ────────────────────
    Final (last 100):  {np.mean(rewards[-100:]):.2f}
    
    EXPLORATION
    ────────────────────
    Current ε:         {agent.epsilon:.4f}
    
    MEMORY
    ────────────────────
    Buffer size:       {len(agent.memory)}
    """

    ax3.text(
        0.1,
        0.95,
        metrics_text,
        transform=ax3.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    import tqdm

    instance_folder = "InstancesEx1/"

    agent = train(
        instance_folder=instance_folder,
        num_episodes=1000,
        save_path="simple_dqn.keras",
    )
    # plot_rl_diagnostics(agent, save_path='diagnostics.png')
    # plot_profit_analysis(agent, save_path='profit_analysis.png')
    # plot_training_summary(agent, save_path='summary.png')
