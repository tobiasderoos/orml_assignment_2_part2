import os
import pickle
import yaml
import numpy as np
import tqdm
from gurobipy import GRB
import time


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

    def __init__(
        self,
        instance_folder,
        reset_params=False,
        model_name="qlearning_model.pkl",
    ):
        self.reset_params = reset_params
        self.instance_folder = instance_folder
        self.model_name = model_name

        # Filenames derived automatically
        self.model_pkl = f"{self.model_name}.pkl"
        self.model_yaml = f"{self.model_name}.yaml"

        # Actions: stopping criterion values (number of items to select in greedy)
        # Here: 5, 10, ..., 95
        self.actions = list(range(15, 120, 15))

        # States: based on instance characteristics
        self.n_states = 9

        # Q-table: states x actions
        self.q_table = np.zeros((self.n_states, len(self.actions)))

        # Q-learning / bandit parameters
        self.alpha = 0.1  # Learning rate
        self.epsilon = 1.0  # Exploration rate (starts high)
        self.epsilon_decay = 0.9975
        self.epsilon_min = 0.05

        # Load
        self.load_if_exists()

        # Reset

        # Caches
        self.full_ilp_cache = {}
        self.reduced_ilp_cache = {}

    def _save(self):
        with open(self.model_pkl, "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "actions": self.actions,
                    "n_states": self.n_states,
                },
                f,
            )
        print(f"Model saved to {self.model_pkl}")

    def _save_params(self, current_episode=0):
        with open(self.model_yaml, "w") as f:
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
        print(f"Parameters saved to {self.model_yaml}")

    def _load(self):
        with open(self.model_pkl, "r") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.actions = data["actions"]
        self.n_states = data["n_states"]
        print(f"Loaded model from {self.model_pkl}")

    def _load_params(self):
        with open(self.model_yaml, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.alpha = data["alpha"]
        self.epsilon = data["epsilon"]
        self.epsilon_decay = data["epsilon_decay"]
        self.epsilon_min = data["epsilon_min"]
        print(f"Loaded parameters from {self.model_yaml}")

    def load_if_exists(self):
        if os.path.exists(self.model_pkl):
            self._load()
        if (not self.reset_params) and os.path.exists(self.model_yaml):
            self._load_params()

    @staticmethod
    def compute_frac_high_state(profits, weights):
        ratios = [profits[i] / weights[i] for i in range(len(weights))]
        n = len(weights)
        avg_ratio = sum(ratios) / n

        if avg_ratio < 2.38:
            ratio_bin = 0
        elif avg_ratio < 2.90:
            ratio_bin = 1
        else:
            ratio_bin = 2
        return avg_ratio, ratio_bin

    @staticmethod
    def compute_theoretical_capacity(weights, capacity):
        avg_weight = sum(weights) / len(weights)
        theoretical_capacity = capacity / avg_weight

        if theoretical_capacity < 95.06:
            return theoretical_capacity, 0
        elif theoretical_capacity < 105.36:
            return theoretical_capacity, 1
        else:
            return theoretical_capacity, 2

    @staticmethod
    def calculate_state(ratio_bin, capacity_bin):
        return ratio_bin * 3 + capacity_bin

    def get_state(self, weights, profits, capacity):
        _, ratio_bin = self.compute_frac_high_state(profits, weights)
        _, capacity_bin = self.compute_theoretical_capacity(weights, capacity)
        state = self.calculate_state(ratio_bin, capacity_bin)
        return state

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
    def train(self, n_episodes=100):
        instance_files = [
            f for f in os.listdir(self.instance_folder) if f.endswith(".txt")
        ]

        print(
            f"Training Q-learning on {len(instance_files)} instances for {n_episodes} episodes..."
        )
        # TQDM
        for episode in tqdm.tqdm(range(n_episodes)):
            fname = np.random.choice(instance_files)
            filepath = os.path.join(self.instance_folder, fname)

            # Load instance
            n, capacity, weights, quad = read_instance(filepath)
            profits = [quad[i][i] for i in range(n)]

            state = self.get_state(weights, profits, capacity)

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
            ilp_obj_val, _, status = self.solve_full_ilp_cache(
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
                self._save()
                self._save_params(current_episode=episode + 1)

            print("\nTraining completed")
            print("\nLearned Q-table:")
            print(self.q_table)


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from greedy import (
        read_instance,
        greedy_qkp,
        compute_profit,
        solve_reduced_ilp,
        solve_ilp,
    )

    # Create and train the Q-learning agent

    agent = QLearning(
        instance_folder="InstancesEx1_200/",
        reset_params=True,
        model_name="qlearning_25_states",
    )

    episodes = 100
    # Train on instances
    agent.train(n_episodes=episodes)

    # Save the trained model
    agent._save()
    agent._save_params(current_episode=episodes)

    print("\n" + "-" * 60)
    print("Training complete - Model saved as")
    print("-" * 60)
