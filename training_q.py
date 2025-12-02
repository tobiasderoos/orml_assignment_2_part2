import os
import pickle
import numpy as np
from gurobipy import Model, GRB

from exc_1_greedy import (
    read_instance,
    greedy_qkp,
    solve_reduced_ilp,
)


# ---------------------------------------------------------------------
# Q-Learning Algorithm
# ---------------------------------------------------------------------
class QLearning:
    """
    Q-Learning hyper-heuristic to learn the best stopping criterion.

    States: Instance characteristics (theoretical capacity)
    Actions: Number of items to add in greedy phase (stopping criterion)
    Reward: Quality of solution from reduced ILP
    """

    def __init__(self, n_items=200):
        # Actions: stopping criterion values (number of items to select in greedy)
        self.actions = list(
            range(5, 51, 5)
        )  # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

        # States: based on instance characteristics
        self.n_states = 5

        # Q-table: states x actions
        self.q_table = np.zeros(
            (self.n_states, len(self.actions))
        )

        # Q-learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate (starts high)
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def get_state(self, weights, capacity):
        """
        Map instance characteristics to a discrete state.
        States represent how many items could theoretically fit.
        This helps determine appropriate stopping threshold k.
        """
        # Estimate: how many average-sized items fit?
        avg_weight = sum(weights) / len(weights)
        theoretical_capacity = capacity / avg_weight

        # Discretize into states based on theoretical capacity
        if theoretical_capacity < 86:
            return 0
        elif theoretical_capacity < 93:
            return 1
        elif theoretical_capacity < 99:
            return 2
        elif theoretical_capacity < 107:
            return 3
        else:
            return 4

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(len(self.actions))
        else:
            # Exploit: best action
            return np.argmax(self.q_table[state])

    def update_q_table(
        self, state, action_idx, reward, next_state
    ):
        """
        Q-learning update rule.
        """
        best_next_action = np.max(self.q_table[next_state])
        current_q = self.q_table[state, action_idx]

        # Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        self.q_table[state, action_idx] = (
            current_q
            + self.alpha
            * (
                reward
                + self.gamma * best_next_action
                - current_q
            )
        )

    def train(self, instance_folder, n_episodes=100):
        """
        Train the Q-learning agent on instances.
        """
        instance_files = [
            f
            for f in os.listdir(instance_folder)
            if f.endswith(".txt")
        ]

        print(
            f"Training Q-learning on {len(instance_files)} instances for {n_episodes} episodes..."
        )

        for episode in range(n_episodes):
            # Randomly select an instance
            fname = np.random.choice(instance_files)
            filepath = os.path.join(instance_folder, fname)

            # Load instance
            n, capacity, weights, quad = read_instance(
                filepath
            )
            profits = [quad[i][i] for i in range(n)]

            # Get state based on instance characteristics
            state = self.get_state(weights, capacity)

            # Choose action (stopping criterion)
            action_idx = self.choose_action(state)
            stopping_criterion = self.actions[action_idx]

            # Run greedy with this stopping criterion
            selected = greedy_qkp(
                weights,
                profits,
                quad,
                capacity,
                stopping_criterion,
            )

            # For episodic task, next_state = terminal (use same state, but doesn't matter much)
            next_state = state

            # Solve reduced ILP
            _, _, ilp_profit = solve_reduced_ilp(
                weights, profits, quad, capacity, selected
            )

            # Normalize reward (divide by a scaling factor for stability)
            reward = ilp_profit / 10000.0

            # Update Q-table
            self.update_q_table(
                state, action_idx, reward, next_state
            )

            # Decay epsilon
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay,
            )

            if (episode + 1) % 10 == 0:
                print(
                    f"Episode {episode + 1}/{n_episodes}, Epsilon: {self.epsilon:.3f}, Reward: {ilp_profit:.0f}, State: {state}, k={stopping_criterion}"
                )

        print("\nTraining completed!")
        print("\nLearned Q-table:")
        print(self.q_table)
        print("\nBest actions per state:")
        for s in range(self.n_states):
            if (
                np.max(self.q_table[s]) > 0
            ):  # Only show states that were visited
                best_action_idx = np.argmax(self.q_table[s])
                visits = np.sum(self.q_table[s] != 0)
                print(
                    f"State {s}: Best k={self.actions[best_action_idx]} (explored {visits}/{len(self.actions)} actions)"
                )
            else:
                print(
                    f"State {s}: Not visited during training"
                )

    def get_best_action(self, state):
        """
        Get the best action for a given state (exploitation only).
        """
        action_idx = np.argmax(self.q_table[state])
        return self.actions[action_idx]

    def save(self, filename):
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

    def load(self, filename):
        """
        Load a trained Q-table.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.q_table = data["q_table"]
            self.actions = data["actions"]
            self.n_states = data["n_states"]
        print(f"Model loaded from {filename}")


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Create and train the Q-learning agent
    agent = QLearning(n_items=200)

    # Train on instances
    agent.train("InstancesEx2_200/", n_episodes=100)

    # Save the trained model
    agent.save("qlearning_model.pkl")

    print("\n" + "=" * 60)
    print(
        "Training complete! Model saved as 'qlearning_model.pkl'"
    )
    print("=" * 60)
