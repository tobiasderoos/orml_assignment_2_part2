import os
import pickle
import yaml
import numpy as np
import time
from greedy import (
    read_instance,
    greedy_qkp,
    compute_profit,
    solve_reduced_ilp,
    solve_ilp,
)
import tqdm
from gurobipy import GRB


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

    def __init__(self, n_items=200):
        # Actions: stopping criterion values (number of items to select in greedy)
        # Here: 5, 10, ..., 95
        self.actions = list(range(5, 100, 5))

        # States: based on instance characteristics
        self.n_states_capacity = 5
        self.n_states_ratio = 3
        self.n_states = self.n_states_capacity * self.n_states_ratio

        # Q-table: states x actions
        self.q_table = np.zeros((self.n_states, len(self.actions)))

        # Q-learning / bandit parameters
        self.alpha = 0.1  # Learning rate
        self.epsilon = 1.0  # Exploration rate (starts high)
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def _save(self, filename):
        """
        Save the trained Q-table.
        """
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "actions": self.actions,
                    "n_states": self.n_states,
                },
                f,
            )
        print(f"Model saved to {filename}")

    def _save_params(self, filename, current_episode=0):
        """
        Save the current model parameters to a file.
        """
        with open(filename, "w") as f:
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

    def _load(self, filename):
        """
        Load a trained Q-table.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.q_table = data["q_table"]
            self.actions = data["actions"]
            self.n_states = data["n_states"]
        print(f"Model loaded from {filename}")

    def _load_params(self, parameters):
        """
        Load model parameters from a file.
        """
        with open(parameters, "rb") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self.alpha = data["alpha"]
            self.epsilon = data["epsilon"]
            self.epsilon_decay = data["epsilon_decay"]
            self.epsilon_min = data["epsilon_min"]

    def get_state(self, profits, weights, capacity):
        """
        Map instance characteristics to a discrete state.
        State = (tightness_bucket, fraction_small_bucket)

        - Tightness bucket: based on theoretical capacity (current approach)
        - Fraction small bucket: % items with w < 25
        """
        n = len(weights)
        avg_weight = sum(weights) / n

        # Tightness state
        theoretical_capacity = capacity / avg_weight

        if theoretical_capacity < 86:
            tight_bin = 0
        elif theoretical_capacity < 93:
            tight_bin = 1
        elif theoretical_capacity < 99:
            tight_bin = 2
        elif theoretical_capacity < 107:
            tight_bin = 3
        else:
            tight_bin = 4

        # profit/weight ratio state

        ratios = [profits[i] / weights[i] for i in range(len(weights))]
        n = len(weights)
        avg_ratio = sum(ratios) / n

        if avg_ratio < 2.0:
            ratio_bin = 0
        elif avg_ratio < 2.6:
            ratio_bin = 1
        else:
            ratio_bin = 2

        # 5 tightness bins × 3 frac_high bins = 15 states
        state_id = ratio_bin * 5 + tight_bin
        return state_id

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

    def compute_reward(self, full_profit, greedy_profit, rilp_profit):
        """
        Compute normalized reward based on:

        - full ILP solution (full_profit)
        - greedy-only solution (greedy_profit)
        - reduced ILP solution (rilp_profit)

        Scaling idea from README:

            full ILP   -> reward = 1
            greedy-only -> reward = 0
            RILP       -> in (0,1): (RILP - greedy) / (full - greedy)
        """
        # If ILP failed, penalize heavily
        if rilp_profit is None:
            return -2.0

        # Normalized between 0 and 1
        normalized = (rilp_profit - greedy_profit) / (
            full_profit - greedy_profit
        )
        return normalized

    def compute_time_penalty(self, time, time_limit=15, beta=0.1):
        """
        Compute penalty based on computation time.
        Longer times yield higher penalties.
        """
        penalty = -beta * (time / time_limit)
        return penalty

    def compute_penalty_model_status(self, status):
        """
        Compute penalty based on whether time limit was reached.
        If time limit reached, return -1, else 0.
        """
        if status == GRB.Status.TIME_LIMIT:
            return -1.0
        elif status == GRB.Status.INFEASIBLE:
            return -1.0
        else:
            return 0.0

    # Training loop
    def train(self, instance_folder, n_episodes=100):
        """
        Train the Q-learning agent (contexåtual bandit) on instances.
        For each episode:
            1. Sample an instance
            2. Compute state from instance characteristics
            3. Pick an action (greedy stopping threshold k)
            4. Run greedy with k -> selected items
            5. Compute:
                - greedy-only profit
                - full ILP profit (no fixed items)
                - reduced ILP profit (fixed greedy items)
            6. Compute reward and penalties from these three values
            7. Update Q-table
        """
        instance_files = [
            f for f in os.listdir(instance_folder) if f.endswith(".txt")
        ]

        print(
            f"Training Q-learning on {len(instance_files)} instances for {n_episodes} episodes..."
        )
        # TQDM
        for episode in tqdm.tqdm(range(n_episodes)):
            # Randomly select an instance
            fname = np.random.choice(instance_files)
            filepath = os.path.join(instance_folder, fname)

            # Load instance
            n, capacity, weights, quad = read_instance(filepath)
            profits = [quad[i][i] for i in range(n)]

            # Get state based on instance characteristics
            state = self.get_state(profits, weights, capacity)

            # Choose action (stopping criterion)
            action_idx = self.choose_action(state)
            stopping_criterion = self.actions[action_idx]

            # Greedy
            selected_greedy = greedy_qkp(
                weights,
                profits,
                capacity,
                stopping_criterion=stopping_criterion,
            )
            greedy_profit = compute_profit(selected_greedy, profits, quad)

            # Full ILP
            ilp_obj_val, ilp_x, ilp_status = solve_ilp(
                weights, profits, quad, capacity
            )

            # Reduced ILP
            start = time.time()
            rilp_obj_val, rilp_x, rilp_status = solve_reduced_ilp(
                weights, profits, quad, capacity, selected_greedy
            )
            end = time.time()
            rilp_time = end - start
            # Rewards and penalties
            reward = self.compute_reward(
                full_profit=ilp_obj_val,
                greedy_profit=greedy_profit,
                rilp_profit=rilp_obj_val,
            )

            penalty_time = self.compute_time_penalty(rilp_time)
            reward += penalty_time

            penalty_status = self.compute_penalty_model_status(rilp_status)
            reward += penalty_status

            self.update_q_table(state, action_idx, reward)

            # Decay epsilon
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay,
            )

            if (episode + 1) % 10 == 0:
                print(
                    f"Episode {episode + 1}/{n_episodes}, "
                    f"Epsilon: {self.epsilon:.3f}, "
                    f"Reward: {reward:.3f}, "
                    f"State: {state}, k={stopping_criterion}"
                )
            # Store model every 10 episodes
            if (episode + 1) % 10 == 0:
                self._save("qlearning_model.pkl")
                self._save_params(
                    "qlearning_params.yaml", current_episode=episode + 1
                )

        print("\nTraining completed")
        print("\nLearned Q-table:")
        print(self.q_table)
        print("\nBest actions per state:")
        for s in range(self.n_states):
            if np.max(self.q_table[s]) > 0:
                best_action_idx = np.argmax(self.q_table[s])
                visits = np.sum(self.q_table[s] != 0)
                print(
                    f"State {s}: Best k={self.actions[best_action_idx]} "
                    f"(explored {visits}/{len(self.actions)} actions)"
                )
            else:
                print(f"State {s}: Not visited during training")


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Create and train the Q-learning agent

    agent = QLearning(n_items=200)

    # Train on instances
    agent.train("InstancesEx1_200/", n_episodes=100)

    # Save the trained model
    agent._save("qlearning_model.pkl")
    agent._save_params("qlearning_params.yaml", current_episode=100)

    print("\n" + "=" * 60)
    print("Training complete! Model saved as 'qlearning_model.pkl'")
    print("=" * 60)
