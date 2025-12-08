import datetime
import os
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tqdm

from e1_greedy import read_instance, compute_profit

from scipy.ndimage import uniform_filter1d


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
        self.step_count = 0

        # Normalization factors
        self.max_w = np.max(self.weights)
        self.max_p = np.max(self.profits)
        self.max_q = np.max(np.triu(self.quad, k=1))

        self.max_gain = np.max(self.profits + np.sum(self.quad, axis=1))

        # Observation space
        obs_size = 11 * self.n + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n)

        # Internal state
        self.selected = None
        self.available = None
        self.remaining_capacity = None
        self.current_weight = None
        self.current_profit = 0.0
        self.current_reward = 0.0

        # Compute greedy profit
        self.greedy_profit, self._marginal_profits = greedy_heuristic(
            self.weights, self.profits, self.quad, self.capacity
        )

    def reset(self, *, seed=None):
        super().reset(seed=seed)

        self.selected = np.zeros(self.n, dtype=np.int32)
        self.available = np.ones(self.n, dtype=bool)
        self.remaining_capacity = self.capacity
        self.current_weight = 0.0
        self.current_profit = 0.0

        return self._get_obs(), {}

    def step(self, action):
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
        # Calculate marginal profit using internal step counter
        marginal = self._marginal_profit(action)
        greedy_marginal = (
            self._marginal_profits[self.step_count]
            if self.step_count < len(self._marginal_profits)
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
        self.available = (
            (self.selected == 0) & (self.weights <= self.remaining_capacity)
        ).astype(np.float32)

        self.step_count += 1
        # Check termination
        if not np.any(self.available):
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

        self.available = ((self.selected == 0) & (fits == 1)).astype(bool)

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

    def _get_q_values(self, state, feasible_mask):
        """Shared logic for getting masked Q-values"""
        q_values = self.online_model.predict(state[np.newaxis, :], verbose=0)[0]
        q_values[~feasible_mask] = -np.inf
        return q_values

    def act(self, state, feasible_mask):
        """Select action using epsilon-greedy (EXPLORATION)"""
        feasible_actions = np.where(feasible_mask)[0]
        if len(feasible_actions) == 0:
            return 0

        # Exploration
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(feasible_actions))

        # Exploitation
        q_values = self._get_q_values(state, feasible_mask)
        return int(np.argmax(q_values))

    def select_greedy_action(self, state, feasible_mask):
        """Select best action (NO EXPLORATION) - used for evaluation"""
        q_values = self._get_q_values(state, feasible_mask)
        return int(np.argmax(q_values))

    def evaluate(self, env):
        """Evaluate on one episode without exploration"""
        state, _ = env.reset()
        total_reward = 0

        while True:
            if not np.any(env.available):
                break

            action = self.select_greedy_action(state, env.available)
            state, reward, terminated, _, info = env.step(action)
            total_reward += reward

            if terminated:
                break

        final_profit = compute_profit(env.selected, env.profits, env.quad)
        optimality_ratio = final_profit / env.greedy_profit

        return total_reward, final_profit, optimality_ratio

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

    def save(self, path):
        """Save only the online model"""
        self.online_model.save(path)

    def load(self, path):
        """Load weights into online model and copy to target"""
        self.online_model = models.load_model(path)
        self.target_model.set_weights(self.online_model.get_weights())


def train_episode(env, agent):
    """Train one episode"""
    state, _ = env.reset()
    total_reward = 0
    episode_q_values = []
    episode_td_errors = []
    steps = 0
    while True:
        # Get feasible actions
        feasible = env.available
        if not np.any(feasible):
            break

        # Select and perform action
        action = agent.act(state, feasible)
        next_state, reward, terminated, _, info = env.step(action)

        # Store experience
        next_feasible = env.available if not terminated else None
        agent.remember(
            state, action, reward, next_state, terminated, next_feasible
        )

        replay_out = agent.replay()
        if replay_out is not None:
            loss, td_errors = replay_out
            episode_td_errors.extend(td_errors)

        loss, td_errors = replay_out if replay_out is not None else (0, [])
        total_reward += reward

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
    instance_files,
    num_episodes=1000,
    save_path="exc2.keras",
    max_instances=None,
    print_interval=10,
):
    """Main training loop"""

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


def smooth_ema(x, alpha=0.1):
    """Exponential moving average smoothing."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def plot_rl_diagnostics(agent, alpha=0.1, save_path=None):
    """
    Simple RL training diagnostics for DQN.

    Plots:
    1. optimality ratio (profit / greedy)
    2. rewards over episodes
    3. training loss
    4. q-value statistics
    5. epsilon decay
    6. episode lengths
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    episodes = np.arange(len(agent.all_episode_rewards))

    # -------------------------------
    # 1. optimality (profit vs greedy)
    # -------------------------------
    ax1 = axes[0, 0]
    opt = np.array(agent.all_episode_optimality)

    ax1.plot(episodes, opt, alpha=0.3, color="blue", label="raw")
    ax1.plot(
        smooth_ema(opt, alpha), color="blue", linewidth=2, label="smoothed"
    )

    ax1.axhline(1.0, color="red", linestyle="--", label="greedy = 1")

    ax1.set_xlabel("episode")
    ax1.set_ylabel("optimality")
    ax1.set_title("optimality ratio (profit / greedy)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # -------------------------------
    # 2. rewards
    # -------------------------------
    ax2 = axes[0, 1]
    rewards = np.array(agent.all_episode_rewards)

    ax2.plot(episodes, rewards, alpha=0.3, color="green", label="raw")
    ax2.plot(
        smooth_ema(rewards, alpha), color="green", linewidth=2, label="smoothed"
    )

    ax2.set_xlabel("episode")
    ax2.set_ylabel("reward")
    ax2.set_title("episode rewards")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # -------------------------------
    # 3. loss
    # -------------------------------
    ax3 = axes[1, 0]

    if agent.losses:
        losses = np.array(agent.losses)
        ax3.plot(losses, alpha=0.3, color="orange", label="raw")
        ax3.plot(
            smooth_ema(losses, alpha),
            color="orange",
            linewidth=2,
            label="smoothed",
        )

        ax3.set_yscale("log")

        ax3.set_title("training loss")
        ax3.set_xlabel("training step")
        ax3.set_ylabel("loss")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "no loss data", ha="center", va="center")
        ax3.set_title("training loss")

    # -------------------------------
    # 4. q-values
    # -------------------------------
    ax4 = axes[1, 1]

    if agent.q_value_history:
        q_means = [q["mean"] for q in agent.q_value_history]
        q_maxs = [q["max"] for q in agent.q_value_history]
        q_mins = [q["min"] for q in agent.q_value_history]

        steps = np.arange(len(q_means))

        ax4.plot(steps, q_means, label="mean", color="blue")
        ax4.fill_between(steps, q_mins, q_maxs, alpha=0.2, color="blue")

        ax4.set_title("q-value stats")
        ax4.set_xlabel("training step")
        ax4.set_ylabel("q-value")
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "no q-value data", ha="center", va="center")
        ax4.set_title("q-value stats")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(
            save_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight"
        )
    plt.show()


def load_agent(model_path, example_env):
    """Recreate an agent with the right architecture and load weights."""
    agent = QAgent(
        state_size=example_env.observation_space.shape[0],
        action_size=example_env.action_space.n,
    )
    agent.load(model_path)
    print(f"Loaded agent from {model_path}")
    return agent


def evaluate_agent_on_instances(agent, instance_files):
    results = []
    profits = []
    optimalities = []

    for idx, fname in enumerate(instance_files):
        n, cap, w, q = read_instance(fname)
        p = [q[i][i] for i in range(n)]
        env = QEnv(w, p, q, cap, instance_id=idx)

        _, profit, optimality = agent.evaluate(env)
        results.append((idx, profit, optimality))
        profits.append(profit)
        optimalities.append(optimality)

        print(
            f"Instance {idx}: Profit={profit:.1f}, Optimality={optimality:.3f}"
        )

    # Averages
    avg_profit = sum(profits) / len(profits) if profits else float("nan")
    avg_optimality = (
        sum(optimalities) / len(optimalities) if optimalities else float("nan")
    )

    print("\nSummary:")
    print(f"Average Profit: {avg_profit:.1f}")
    print(f"Average Optimality: {avg_optimality:.3f}")

    return {
        "results": results,
        "avg_profit": avg_profit,
        "avg_optimality": avg_optimality,
    }


if __name__ == "__main__":
    instance_folder = "InstancesEx2/"
    instance_files = sorted(
        os.path.join(instance_folder, f)
        for f in os.listdir(instance_folder)
        if os.path.isfile(os.path.join(instance_folder, f))
    )

    # Train
    agent = train(
        instance_files=instance_files,
        num_episodes=50,
        save_path="exc_2_model/test_model.keras",
    )
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        plot_rl_diagnostics(
            agent,
            alpha=alpha,
            save_path=f"exc_2_plots/training_diagnostics_alpha{alpha}.png",
        )

    # Rebuild agent for testing loading
    n, cap, w, q = read_instance(instance_files[0])
    p = [q[i][i] for i in range(n)]
    example_env = QEnv(w, p, q, cap, instance_id=0)

    # Load model
    loaded_agent = load_agent("exc_2_model/test_model.keras", example_env)

    # Evaluate over all instances
    evaluate_agent_on_instances(loaded_agent, instance_files)
