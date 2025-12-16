# ---------------------------------------------------------------------
# Read the instance
# ---------------------------------------------------------------------
def read_instance(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    c = int(lines[1])
    w = list(map(int, lines[2].split()))
    p_lines = lines[3:]
    p = [list(map(int, row.split())) for row in p_lines]
    return n, c, w, p


# ---------------------------------------------------------------------
# Run the hyper-heuristic -- here something trivial
# ---------------------------------------------------------------------
def run_hyperheuristic(instance_file, agent, env_creator):
    n, c, w, p = read_instance(instance_file)
    profits = [p[i][i] for i in range(n)]

    env = fe.extract(n, weights, profits, p, c)
    idx, _ = agent.act(env, train=False)
    stopping = agent.actions[idx]

    sel = greedy_qkp(w, profits, p, c, stopping_criterion=stopping)
    rl_profit, _, _ = solve_reduced_ilp(w, profits, p, c, sel)

    return rl_profit


# ---------------------------------------------------------------------
# Evaluate the subset of 20 instances and print the results in a file
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from e1_training import DQNAgent, FeatureExtractor
    from e1_testing import solve_reduced_ilp, greedy_qkp

    NUM_INSTANCES = 20
    RESULTS_FILE = "results1.txt"

    # Initialize agent
    fe = FeatureExtractor()
    agent = DQNAgent(feature_dim=fe.feature_dim)
    all_results = []

    results = []

    for i in range(NUM_INSTANCES):
        instance_file = f"InstancesEx1/instance{i}.txt"
        value = run_hyperheuristic(instance_file, agent, fe)
        results.append(value)

    with open(RESULTS_FILE, "w") as f:
        for val in results:
            f.write(f"{val}\n")
