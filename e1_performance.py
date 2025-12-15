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
def run_hyperheuristic(instance_file, agent):
    n, c, w, p = read_instance(instance_file)
    profits = [p[i][i] for i in range(n)]
    rl_result = agent.evaluate_instance(n, c, w, p)

    return rl_result["rilp_profit"]


# ---------------------------------------------------------------------
# Evaluate the subset of 20 instances and print the results in a file
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from e1_training import QLearning

    NUM_INSTANCES = 20
    RESULTS_FILE = "results1.txt"

    # Initialize agent
    agent = QLearning(
        instance_files=[],  # leave empty, won't train here
        reset_params=False,  # don't reset parameters
        model_name="exc_1_model/qkeras_model",
    )
    all_results = []

    results = []

    for i in range(NUM_INSTANCES):
        instance_file = f"InstancesEx1/instance{i}.txt"
        value = run_hyperheuristic(instance_file, agent)
        results.append(value)

    with open(RESULTS_FILE, "w") as f:
        for val in results:
            f.write(f"{val}\n")
