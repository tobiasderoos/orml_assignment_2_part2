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
        self.actions = list(range(15, 120, 15))

        # States: based on instance characteristics
        self.n_states = 5

        # Q-table: states x actions
        self.q_table = np.zeros((self.n_states, len(self.actions)))

        # Q-learning / bandit parameters
        self.alpha = 0.1  # Learning rate
        self.epsilon = 1.0  # Exploration rate (starts high)
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        # Load
        self.load_if_exists()
        # Caches
        self.full_ilp_cache = {}
        self.reduced_ilp_cache = {}

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

    def load_if_exists(self):
        """Load Q-table + params if available."""
        if os.path.exists("qlearning_model.pkl"):
            self._load("qlearning_model.pkl")

        if os.path.exists("qlearning_params.yaml"):
            self._load_params("qlearning_params.yaml")

    def get_state(self, weights, profits):
        ratios = [profits[i] / weights[i] for i in range(len(weights))]
        avg_ratio = sum(ratios) / len(ratios)

        t1 = 2.168
        t2 = 2.440
        t3 = 2.712
        t4 = 3.796

        if avg_ratio < t1:
            return 0
        elif avg_ratio < t2:
            return 1
        elif avg_ratio < t3:
            return 2
        elif avg_ratio < t4:
            return 3
        else:
            return 4

    def solve_full_ilp_cache(self, fname, weights, profits, quad, capacity):
        """
        Solve full ILP and cache results.
        """
        if fname in self.full_ilp_cache:
            return self.full_ilp_cache[fname]
        else:
            obj_val, x, status = solve_ilp(weights, profits, quad, capacity)
            self.full_ilp_cache[fname] = (obj_val, x, status)
            return obj_val, x, status

    def solve_rilp_cache(
        self,
        fname,
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
        key = (fname, stopping_criterion)

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

    def compute_reward(self, full_profit, greedy_profit, rilp_profit):
        if rilp_profit is None:
            return -2.0
        else:
            return (rilp_profit - greedy_profit) / (full_profit - greedy_profit)

    def compute_time_penalty(self, t, time_limit=15, beta=0.1):
        return -beta * (t / time_limit)

    def compute_penalty_model_status(self, status):
        if status in [GRB.Status.TIME_LIMIT, GRB.Status.INFEASIBLE]:
            return -1.0
        return 0.0

    # Training loop
    def train(self, instance_folder, n_episodes=100):
        instance_files = [
            f for f in os.listdir(instance_folder) if f.endswith(".txt")
        ]

        print(
            f"Training Q-learning on {len(instance_files)} instances for {n_episodes} episodes..."
        )
        # TQDM
        for episode in tqdm.tqdm(range(n_episodes)):
            fname = np.random.choice(instance_files)
            filepath = os.path.join(instance_folder, fname)

            # Load instance
            n, capacity, weights, quad = read_instance(filepath)
            profits = [quad[i][i] for i in range(n)]

            state = self.get_state(profits, weights)

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

            # Full ILP cached
            ilp_obj_val, _, _ = self.solve_full_ilp_cache(
                fname, weights, profits, quad, capacity
            )

            # Reduced ILP
            start = time.time()
            rilp_obj_val, _, rilp_status = self.solve_rilp_cache(
                fname,
                weights,
                profits,
                quad,
                capacity,
                selected_greedy,
                stopping_criterion,
            )
            end = time.time()

            # Rewards and penalties
            reward = self.compute_reward(
                full_profit=ilp_obj_val,
                greedy_profit=greedy_profit,
                rilp_profit=rilp_obj_val,
            )

            reward += self.compute_time_penalty(end - start)
            reward += self.compute_penalty_model_status(rilp_status)

            self.update_q_table(state, action_idx, reward)

            # Decay epsilon
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay,
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


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Create and train the Q-learning agent

    agent = QLearning(n_items=30)

    # Train on instances
    agent.train("InstancesEx1_200/", n_episodes=100)

    # Save the trained model
    agent._save("qlearning_model.pkl")
    agent._save_params("qlearning_params.yaml", current_episode=100)

    print("\n" + "=" * 60)
    print("Training complete! Model saved as 'qlearning_model.pkl'")
    print("=" * 60)
